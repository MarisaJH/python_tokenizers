from transformers.tokenization_utils import ExtensionsTrie
from tokenizers.models import Model
from tokenizers import AddedToken
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple, Iterator, Optional, Set
from functools import lru_cache
from tqdm import tqdm
import json
from pathlib import Path

import math
from collections import defaultdict
import numpy as np
from scipy.special import digamma
import unittest
import pickle

from .lattice import Lattice
import suffix_array_rs
K_UNK_PENALTY = 10.0

class Token:
    def __init__(self, id: int, text: str, offsets: Tuple[int, int]):
        self.id = id
        self.text = text
        self.offsets = offsets

    def __repr__(self):
        return f"Token(id={self.id}, text={self.text}, offsets={self.offsets})"
    
class UnigramTrainerError(Exception):
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)

class VocabularyTooSmallError(UnigramTrainerError):
    def __init__(self, message="The vocabulary is not large enough to contain all chars"):
        self.message = message
        super().__init__(self.message)

class UnigramTrainer():
    def __init__(
        self,
        show_progress: bool = True,
        vocab_size: int = 8000,
        n_sub_iterations: int = 2,
        shrinking_factor: float = 0.75,
        special_tokens: List[AddedToken] = None,
        initial_alphabet: Set[str] = None,
        unk_token: Optional[str] = None,
        max_piece_length: int = 16,
        seed_size: int = 1_000_000,
        words: Dict[str, int] = None,
        num_threads: int = 1,
    ):
        self.show_progress = show_progress
        self.vocab_size = vocab_size
        self.n_sub_iterations = n_sub_iterations
        self.shrinking_factor = shrinking_factor
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.initial_alphabet = initial_alphabet if initial_alphabet is not None else set()
        self.unk_token = unk_token
        self.max_piece_length = max_piece_length
        self.seed_size = seed_size
        self.words = words if words is not None else {}
        self.num_threads = num_threads

        # if special_tokens aren't set to special=True, then set it
        for token in self.special_tokens:
            if not token.special:
                token.special = True

    def make_seed_sentence_pieces(
        self,
        sentences: List[Tuple[str, int]],
        progress: Optional[object] = None  # Placeholder for any progress bar
    ) -> List[Tuple[str, float]]:
        """
        Creates an initial set of seed sentence pieces from given sentences.
        """
        if progress is not None: progress.set_description("Counting characters")
        
        # 1. Concatenate sentences into a flat string, separated by a null character
        #total = sum(len(s) for s, _ in sentences) + len(sentences)
        flat_string = ''.join(s + '\0' for s, _ in sentences)
        c_sentence_boundary = '\0'
        #flat_string = c_sentence_boundary.join(f" {c_sentence_boundary}".join(sentence.split()) for sentence, _ in sentences)
        all_chars: Dict[str, int] = defaultdict(int)
        
        #print('flat str:', flat_string)

        # Count character frequencies, ignoring the sentence boundary character
        for string, count in sentences:
            for char in string:
                if char != c_sentence_boundary:
                    all_chars[char] += count

        #print('all chars:', all_chars)
        if progress is not None: 
            progress.update(1)
            progress.set_description("Generating suffix array")

        # 2. Generate suffix array for `flat_string`
        #suffix_array = SuffixArray(flat_string).suffix_array()
        #suffix_array = self.simple_suffix_array_with_frequencies(flat_string, separator=c_sentence_boundary)
        suffix_array = suffix_array_rs.suffix(flat_string)
        if progress is not None:
            progress.update(1)
            progress.set_description("Collecting seed sentence pieces")
        #print('suffix array:', suffix_array)
        # 3. Collect basic characters as seed sentence pieces
        seed_sentencepieces = [(char, count) for char, count in all_chars.items()]
        seed_sentencepieces.sort(reverse=True, key=lambda x: x[1])  # Sort by frequency
        #print('seed sentence pieces:', seed_sentencepieces)
        if progress is not None:
            progress.update(1)
            progress.set_description("Scoring substring candidates")

        # 4. Filter and score substrings in the suffix array
        substr_index = []
        for substring, freq in suffix_array:
            if len(substring) <= 1 or c_sentence_boundary in substring:
                continue
            if not self.is_valid_sentencepiece(substring):
                continue
            score = freq * len(substring)
            substr_index.append((score, substring))

        # Sort substrings by score in descending order
        substr_index.sort(reverse=True, key=lambda x: x[0])
        #print('sorted substr index:', substr_index)

        if progress is not None:
            progress.update(1)
            progress.set_description("Adding high-scoring substrings to seed pieces")
        
        # 5. Add high-scoring substrings to seed pieces until reaching seed_size
        for score, substring in substr_index:
            if len(seed_sentencepieces) >= self.seed_size:
                break
            #print(f"Adding to seed pieces: {substring}, initial score: {score}")
            seed_sentencepieces.append((substring, score))

        #print("Before to_log_prob:", seed_sentencepieces)  # Show first 10
        self.to_log_prob(seed_sentencepieces)
        #print("After to_log_prob:", seed_sentencepieces)  # Show first 10

        if progress is not None:
            progress.update(1)
            progress.set_description("Seed sentence pieces collected")
            # will be closed in the main training loop
        #print('final seed sentence pieces:', seed_sentencepieces)
        return seed_sentencepieces

    def is_valid_sentencepiece(self, char_string: str) -> bool:
        """
        Checks if the provided character sequence is a valid sentence piece.
        
        Args:
            char_string (str): The character sequence to validate.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        # Check if the string is non-empty and does not exceed the max piece length
        n = len(char_string)
        if n == 0 or n > self.max_piece_length:
            return False
        return True
    
    def to_log_prob(self, sentence_pieces: List[Tuple[str, int]]):
        """
        Converts frequency counts to log probabilities in-place.
        """
        total = sum(score for _, score in sentence_pieces)
        for i, (piece, count) in enumerate(sentence_pieces):
            sentence_pieces[i] = (piece, math.log(count / total))

    def run_e_step(self, model: "Unigram", sentences: List[Tuple[str, int]]) -> Tuple[float, int, List[float]]:
        """
        E-step of the EM algorithm, calculating the expected counts and the objective.
        
        Args:
            model (Unigram): The current unigram model.
            sentences (List[Tuple[str, int]]): List of sentences and their frequencies.
        
        Returns:
            Tuple[float, int, List[float]]: Tuple containing the total objective, 
                                            total token count, and expected token counts.
        """
        # Calculate total sentence frequencies
        all_sentence_freq = sum(freq for _, freq in sentences)
        chunk_size = max(len(sentences) // self.num_threads, 1)
        
        def process_chunk(sentences_chunk):
            #expected = [0.0] * model.len()
            expected = np.zeros(model.len())
            objs = 0.0
            ntokens = 0
            #print('sentences_chunk:', sentences_chunk)

            for string, freq in sentences_chunk:
                #print('string, freq:', string, freq)
                lattice = Lattice(string, model.bos_id, model.eos_id)
                #print('lattice:', lattice)
                model.populate_nodes(lattice)
                #print('lattice after populate_nodes:', lattice)
                z = lattice.populate_marginal(freq, expected)
                if np.isnan(z):
                    raise ValueError("Likelihood is NAN. Input sentence may be too long.")
                    
                ntokens += len(lattice.viterbi())
                objs -= z / all_sentence_freq

            return objs, ntokens, expected

        # Parallel processing
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        total_objs, total_ntokens, total_expected = 0.0, 0, np.zeros(model.len())

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        for objs, ntokens, expected in results:
            total_objs += objs
            total_ntokens += ntokens
            total_expected += expected

        return total_objs, total_ntokens, total_expected.tolist()

    def run_m_step(self, pieces, expected):
        if len(pieces) != len(expected):
            raise ValueError(f"Expected pieces and expected frequencies to have the same length, but got {len(pieces)} and {len(expected)}")

        new_pieces = []
        sum_freq = 0.0
        expected_frequency_threshold = 0.5
        #print('M step:')
        for i, (freq, (piece, _score)) in enumerate(zip(expected, pieces)):
            # Always keep unk
            #print(f"piece: {piece}, freq: {freq}, score: {_score}")
            if i == 0:
                new_pieces.append((piece, 0.0))
                continue
            if freq < expected_frequency_threshold:
                continue
            new_pieces.append((piece, freq))
            sum_freq += freq

        # Apply Bayesianified EM algorithm adjustments
        logsum = digamma(sum_freq)
        new_pieces = [(s, digamma(c) - logsum) for s, c in new_pieces]
        # preserve the score of the UNK token as 0.0
        new_pieces[0] = (new_pieces[0][0], 0.0)
        
        return new_pieces

    def prune_sentence_pieces(
        self,
        model: "Unigram",
        pieces: List[Tuple[str, float]],  # List of sentence pieces with scores
        sentences: List[Tuple[str, int]],  # List of sentences and their frequencies
    ) -> List[Tuple[str, float]]:
        # Step 1: Initialize flags and alternative mappings
        always_keep = [True] * len(pieces)
        alternatives = [[] for _ in range(len(pieces))]

        bos_id = len(pieces) + 1
        eos_id = len(pieces) + 2

        # Step 2: For each sentence piece, find alternative segmentations
        for id, (token, score) in enumerate(pieces):
            if id == 0:  # Skip the unknown token
                always_keep[id] = False
                continue

            lattice = Lattice(token, bos_id, eos_id)
            model.populate_nodes(lattice)
            nbests = lattice.nbest(2, return_vocab_ids=True)

            if len(nbests) == 1:
                always_keep[id] = True
            elif len(nbests[0]) >= 2:
                always_keep[id] = False
            elif len(nbests[0]) == 1:
                always_keep[id] = True
                #alternatives[id] = [node['id'] for node in nbests[1]]
                alternatives[id] = nbests[1]

        # Step 3: Segment sentences and calculate frequencies
        vsum, freq, inverted = 0.0, np.zeros(len(pieces)), [[] for _ in range(len(pieces))]
        for i, (sentence, count) in enumerate(sentences):
            lattice = Lattice(sentence, bos_id, eos_id)
            model.populate_nodes(lattice)
            vsum += count
            for node_vocab_id in lattice.viterbi(return_vocab_ids=True):
                #piece_id = node['id']
                freq[node_vocab_id] += count
                inverted[node_vocab_id].append(i)

        # Step 4: Prune sentence pieces based on calculated frequencies and likelihood loss
        sum_freq = freq.sum()
        logsum = np.log(sum_freq)
        candidates = []
        new_pieces = [pieces[0]]  # Always keep the unknown token

        for id, (token, score) in enumerate(pieces):
            if id == 0:
                continue
            if freq[id] == 0 and not always_keep[id]:
                continue
            elif not alternatives[id]:
                new_pieces.append((token, score))
            else:
                f = sum(sentences[i][1] for i in inverted[id]) / vsum
                logprob_sp = np.log(freq[id]) - logsum
                logsum_alt = np.log(sum_freq + freq[id] * (len(alternatives[id]) - 1))

                logprob_alt = sum(
                    np.log(freq[alt_id] + freq[id]) - logsum_alt
                    for alt_id in alternatives[id]
                )

                loss = f * (logprob_sp - logprob_alt)
                if not np.isnan(loss):
                    candidates.append((id, loss))

        # Step 5: Sort and retain top candidates based on shrinking factor and target vocab size
        pruned_size = max(int(len(pieces) * self.shrinking_factor), int(self.vocab_size * 1.1))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for id, _ in candidates:
            if len(new_pieces) == pruned_size:
                break
            new_pieces.append(pieces[id])

        return new_pieces

    def finalize(self, model, required_chars):
        min_score_penalty = 0.0
        min_score_penalty_delta = 0.0001
        pieces = []
        inserted = set()

        # Avoid including training <UNK>, which is always the first token
        inserted.add("<unk>") 
        inserted.add(self.unk_token)

        # Gather existing pieces
        existing_pieces = {token: score for token, score in model}

        # Ensure all required chars are in pieces
        for char in required_chars:
            if char in existing_pieces:
                pieces.append((char, existing_pieces[char]))
            else:
                score = model.min_score + min_score_penalty
                pieces.append((char, score))
                min_score_penalty += min_score_penalty_delta
            inserted.add(char)

        # Check if we need to add <UNK>
        if self.unk_token:
            unk_id = next((i for i, t in enumerate(self.special_tokens) if t.content == self.unk_token), None)
            need_add_unk = unk_id is None
            unk_id = unk_id if unk_id is not None else 0
        else:
            unk_id = None
            need_add_unk = False
        #print('need_add_unk:', need_add_unk)
        #print('self.unk_token:', self.unk_token)

        vocab_size_without_special_tokens = self.vocab_size - len(self.special_tokens) - (1 if need_add_unk else 0)

        # Add other tokens from the model
        for token, score in model:
            if token in inserted:
                #print(f"token already in inserted: {token}")
                continue
            pieces.append((token, 0.0 if score is None else score))
            inserted.add(token)

            if len(pieces) >= vocab_size_without_special_tokens:
                break

        # Sort by descending score
        pieces.sort(key=lambda x: -x[1])

        # Add necessary special tokens
        special_tokens = [(t.content, 0.0) for t in self.special_tokens]
        if need_add_unk:
            special_tokens.insert(0, (self.unk_token, 0.0))

        # update model.unk_token if it's different from the default

        # Construct and return the finalized Unigram model
        return Unigram(special_tokens + pieces, unk_id, model.byte_fallback)

    def required_chars(self, word_counts: List[Tuple[str, int]]) -> Set[str]:
        """
        Gather all unique characters from the input sentences and
        initial alphabet, and return as a set of strings.
        """
        # Flatten characters from sentences and add any initial alphabet characters
        chars = set(c for s, _ in word_counts for c in s)
        chars.update(str(c) for c in self.initial_alphabet)
        
        return chars
    
    def do_train(self, sentences: List[Tuple[str, int]], model: "Unigram") -> List[AddedToken]:
        #progress = self.setup_progress()
        #progress_overall = tqdm()

        # 1. Compute frequent substrings
        #self.update_progress(progress, len(sentences), "Suffix array seeds")
        progress_suffix = tqdm(5, desc="Suffix array seeds", disable=not self.show_progress)
        pieces = [(self.unk_token, 0.0)]  # Initialize with the UNK token
        pieces.extend(self.make_seed_sentence_pieces(sentences, progress_suffix))
        #self.finalize_progress(progress, len(sentences))
        progress_suffix.close()

        # Log useful information
        print(f"Using {len(pieces)} pieces on {len(sentences)} sentences for EM training")

        desired_vocab_size = int(self.vocab_size * 1.1)  # * 1.1

        # 2. Run E-M Loops to fine-grain the pieces
        '''
        expected_loops = int(
            ((desired_vocab_size).bit_length() - (len(pieces)).bit_length()) /
            (self.shrinking_factor.bit_length())
        ) + 1 
        '''
         # Calculate expected loops using log base calculation instead of bit_length
        if self.shrinking_factor > 0:
            expected_loops = int(
                math.log(desired_vocab_size / len(pieces)) / 
                math.log(1 / self.shrinking_factor)
            ) + 1
        else:
            expected_loops = 1  # fallback if shrinking_factor is 0

        expected_updates = expected_loops * self.n_sub_iterations

        progress_em = tqdm(total=expected_updates, desc="EM training", disable=not self.show_progress)
        #self.update_progress(progress, expected_updates, "EM training")
        required_chars = self.required_chars(sentences)

        if len(required_chars) > self.vocab_size:
            raise UnigramTrainerError("Vocabulary too small")

        new_model = Unigram(pieces.copy(), 0, False)

        while True:
            # Sub-EM iteration
            for _ in range(self.n_sub_iterations):
                # E step
                objective, num_tokens, expected = self.run_e_step(new_model, sentences)

                # M step
                pieces = self.run_m_step(pieces, expected)
                new_model = Unigram(pieces.copy(), 0, False)

                # Log useful information for debugging
                print(f"Em iter={_} size={len(new_model)} obj={objective} num_tokens={num_tokens}")

                if progress_em is not None:
                    progress_em.update(1)

            # Stops the iteration when the size of sentences reaches the desired symbol size
            if len(pieces) <= desired_vocab_size:
                break

            # Prunes pieces
            pieces = self.prune_sentence_pieces(new_model, pieces, sentences)
            new_model = Unigram(pieces.copy(), 0, False)

        #self.finalize_progress(progress, expected_updates)
        # Finalize the progress bar
        if progress_em is not None:
            # If actual iterations might not match expected_updates, ensure it’s marked complete
            if progress_em.n != expected_updates:
                progress_em.n = expected_updates
                progress_em.refresh()
            progress_em.close()

        # Finally, adjust the size of sentence pieces to be |vocab_size|
        model.update(self.finalize(new_model, required_chars))

        return self.special_tokens.copy()
    
    def train(self, model: "Unigram") -> List[AddedToken]:
        """
        Train the model using word counts (self.words, set with feed())
        
        Args:
            words: Dictionary mapping words to their frequencies
            
        Returns:
            List of special tokens
        """

        # Now call do_train with the processed data
        sentences = [(word, count) for word, count in self.words.items()]
        if len(sentences) == 0:
            raise ValueError("No training data provided")
            
        return self.do_train(sentences, model)

    def should_show_progress(self) -> bool:
        return self.show_progress

    '''
    def feed(
        self, 
        iterator: Iterator[str], 
        process: Callable[[str], List[str]] 
    ) -> None:
        # Initialize an empty dictionary to hold the words and their counts
        words_count: Dict[str, int] = {}

        for sequence in iterator:
            words = process(sequence)
            for word in words:
                # Increment the count for each word
                words_count[word] = words_count.get(word, 0) + 1
        
        # Update the words dictionary in the trainer
        self.words = words_count
    '''
    
class Unigram(Model):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(
        self,
        vocab: List[Tuple[str, float]] = None,  # List of (token, score) tuples
        unk_id: Optional[int] = None,
        byte_fallback: bool = False
    ):
        """
        Initialize the Unigram model with a given vocabulary.
        
        Args:
        - `vocab`: A list of (token, score) tuples, where `score` is typically a log probability.
        - `unk_id`: The index of the unknown token within `vocab`. Must be provided and within bounds.
        - `byte_fallback`: Whether to enable byte fallback for unknown characters.

        Raises:
        - ValueError if `vocab` is empty or if `unk_id` is invalid.
        """
        super().__init__()

        # Default initialization with <unk> token if no vocab provided
        if vocab is None:
            vocab = [("<unk>", 0.0)]
            unk_id = 0
            byte_fallback = False
        
        # Ensure that the vocabulary is not empty
        if not vocab:
            raise ValueError("Vocabulary cannot be empty and must contain an `<unk>` token.")

        # Ensure `unk_id` is provided and within the range of vocabulary indices
        if unk_id is None or not (0 <= unk_id < len(vocab)):
            raise ValueError("`unk_id` must be provided and must be within the range of vocabulary indices.")

        # Initialize the token mappings
        self.token_to_ids = {token: idx for idx, (token, _) in enumerate(vocab)}
        self.vocab = vocab  # Store the vocabulary
        self.min_score = min(score for _, score in vocab) if vocab else float('inf')
        #print(f"in init:min_score: {self.min_score}")
        #print(f"in init:unk_id: {unk_id}")
        #print(f"in init:vocab: {vocab}")
        # Set special token IDs
        self.unk_id = unk_id
        self.bos_id = len(vocab) + 1
        self.eos_id = len(vocab) + 2

        # Initialize the Trie and populate it with tokens
        self.trie = ExtensionsTrie()
        for token, _ in vocab:
            self.trie.add(token)

        # Additional flags and configurations
        self.fuse_unk = True  # Default setting
        self.is_optimized = True  # Default setting for optimized flag
        self.byte_fallback = byte_fallback

        self.MAX_LENGTH = 512
        #self.cache = lru_cache(maxsize=self.MAX_LENGTH)(self._cache_lookup)

    def update(self, other: "Unigram"):
        """
        Updates the model's internal state with that of another model.
        
        Args:
        - `other`: A `Unigram` instance to copy state from.
        """
        self.vocab = other.vocab
        self.token_to_ids = other.token_to_ids
        self.trie = other.trie
        self.min_score = other.min_score
        self.unk_id = other.unk_id
        self.bos_id = other.bos_id
        self.eos_id = other.eos_id
        self.fuse_unk = other.fuse_unk
        self.is_optimized = other.is_optimized
        self.byte_fallback = other.byte_fallback
        #self.cache = other.cache
        self.MAX_LENGTH = other.MAX_LENGTH

    @classmethod
    def from_file(cls, model_file: str):
        with open(model_file, "rb") as f:
            return pickle.load(f)

    def _cache_lookup(self, sentence: str) -> Optional[List[str]]:
        """Helper function to use with lru_cache to retrieve a cached encoding."""
        pass

    def common_prefix_search(self, sentence: str, start_pos: int) -> List[str]:
        """
        Searches for all common prefixes in the trie starting from `start_pos`.
        
        Args:
        - `sentence`: The full sentence string.
        - `start_pos`: The position in `sentence` from where to start the search.
        
        Returns:
        - A list of matching prefixes as strings.
        """
        node = self.trie.data  # Start from the root of the Trie
        prefixes = []
        current_prefix = []

        for char in sentence[start_pos:]:
            if char not in node:
                break  # Stop if there’s no further match in the Trie
            node = node[char]
            current_prefix.append(char)

            # Check if this node represents the end of a token
            if self.trie._termination_char in node: 
                # Join chars and add to prefixes
                prefix_str = "".join(current_prefix)
                prefixes.append(prefix_str)

        return prefixes
    
    def populate_nodes(self, lattice: Lattice):
        """
        Populates the `lattice` with nodes by searching for tokens in the trie.
        
        Args:
        - `lattice`: The Lattice object to populate.
        """
        unk_score = self.min_score - K_UNK_PENALTY
        #print(f"UNK score: {unk_score}")
        #print(f"min score: {self.min_score}")
        #print(self.unk_id)
        sentence = lattice.sentence_
        sentence_length = len(sentence)

        begin_pos = 0
        while begin_pos < sentence_length:
            mblen = len(sentence[begin_pos])
            #print(f"\nPosition {begin_pos}, current char: '{sentence[begin_pos]}'")

            has_single_node = False
            
            # Print all possible prefixes at this position
            prefixes = list(self.common_prefix_search(sentence, begin_pos))
            #print(f"Found prefixes at position {begin_pos}: {prefixes}")

            for match_ in prefixes:
                token_length = len(match_)
                token_id = self.token_to_ids[match_]
                score = self.vocab[token_id][1]
                
                #print(f"  Adding token: '{match_}', length: {token_length}, ID: {token_id}, Score: {score}")
                lattice.insert(begin_pos, token_length, score, token_id)

                if not has_single_node and token_length == mblen:
                    has_single_node = True
                    #print(f"  Marked as single node: {match_}")

            if not has_single_node and self.unk_id is not None:
                #print(f"  No single node found at pos {begin_pos}, inserting UNK token")
                #print(f"  UNK ID: {self.unk_id}, Score: {unk_score}")
                lattice.insert(begin_pos, mblen, unk_score, self.unk_id)

            begin_pos += mblen

    def encode(self, sentence: str) -> List[str]:
        """Encodes a sentence, using cache if available."""
        if not sentence:
            return []

        # TODO Attempt to retrieve cached result
        #cached_result = self._cache_lookup(sentence)
        #if cached_result is not None:
        #    return cached_result

        # Encode using unoptimized method
        result = self.encode_unoptimized(sentence)

        # Cache result if sentence is short enough
        #if len(sentence) < self.MAX_LENGTH:
        #    self.cache.cache_clear()  # Clear older cached items
        #    self.cache(sentence)  # Store result in cache

        return result

    def encode_unoptimized(self, sentence: str) -> List[str]:
        """
        Unoptimized encoding that populates the lattice and handles <unk> fusion.
        
        Args:
        - `sentence`: The input string to encode.
        
        Returns:
        - A list of tokens (strings)
        """
        # Initialize the lattice
        lattice = Lattice(sentence, self.bos_id, self.eos_id)
        self.populate_nodes(lattice)

        if self.fuse_unk:
            results = []
            token = ""

            # Iterate over the Viterbi-decoded node IDs
            for node_id in lattice.viterbi():
                vocab_id = lattice.get_vocab_id(node_id)  # Get vocab ID for the node
                piece = lattice.piece(node_id)  # Get the string piece associated with the node

                if vocab_id == self.unk_id:
                    # Append to token if current node is <unk>
                    token += piece
                else:
                    # If there’s a fused <unk> token, add it to results
                    if token:
                        results.append(token)
                        token = ""
                    # Append the current non-<unk> piece
                    results.append(piece)

            # Add any remaining fused <unk> token
            if token:
                results.append(token)

            return results
        else:
            # If not fusing <unk>, simply return the tokens directly
            return lattice.tokens()

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary mapping tokens to their IDs."""
        return self.token_to_ids.copy()

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def tokenize(self, sentence: str) -> List[Token]:
        """
        Tokenizes a sentence into a list of Token objects, with optional byte fallback.
        
        Args:
        - `sentence`: The input string to tokenize.
        
        Returns:
        - A list of `Token` objects with ID, text, and offsets.
        """
        str_tokens = self.encode(sentence)
        tokens = []
        offset = 0

        for string in str_tokens:
            length = len(string)
            offsets = (offset, offset + length)
            
            # Attempt to retrieve the token ID
            token_id = self.token_to_ids.get(string)
            if token_id is None:
                if self.byte_fallback:
                    # Handle byte fallback if the token isn't in the vocabulary
                    byte_tokens = []
                    for byte in string.encode('utf-8'):
                        byte_string = f"<0x{byte:02X}>"
                        byte_id = self.token_to_ids.get(byte_string)
                        if byte_id is not None:
                            byte_token = Token(byte_id, byte_string, (offset, offset + 1))
                            byte_tokens.append(byte_token)
                            offset += 1
                        else:
                            # If any byte token isn't found, treat the whole sequence as <unk>
                            byte_tokens = None
                            break

                    if byte_tokens:
                        tokens.extend(byte_tokens)
                        continue  # Go to the next token after byte fallback

                # Use <unk> ID if no match and no byte fallback is possible
                token_id = self.unk_id
                if token_id is None:
                    raise ValueError("Unknown token encountered, and no <unk> ID is set.")

            # Append the token with its ID and offsets
            tokens.append(Token(token_id, string, offsets))
            offset += length

        return tokens
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Returns the ID of a given token, or None if not found."""
        return self.token_to_ids.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Returns the token corresponding to a given ID, or None if not found."""
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id][0]  # Access the token in the vocab
        return None

    def save(self, folder: Path, name: Optional[str] = None) -> List[Path]:
        """
        Saves the model's vocabulary to a JSON file.
        
        Args:
        - `folder`: The folder to save the file in.
        - `name`: An optional name for the file. If provided, it will be in the format `{name}-unigram.json`.

        Returns:
        - A list containing the saved file's path.
        """
        # Determine the filename
        filename = f"{name}-unigram.json" if name else "unigram.json"
        if folder.endswith('/'): folder = folder[:-1]
        fullpath = f'{folder}/{filename}'
        
        # Write the JSON file directly using self.token_to_ids
        with open(fullpath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

        return [fullpath]

    def get_trainer(self):
        """Returns a default trainer instance for this model."""
        return UnigramTrainer()

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)
    
    def len(self) -> int:
        return len(self.vocab)
    
    def __iter__(self):
        """Makes the model iterable, yielding (token, score) pairs."""
        return iter(self.vocab)

    def iter(self):
        """Alternative iterator method."""
        return iter(self.vocab)
    
    def set_optimized(self, is_optimized: bool):
        self.is_optimized = is_optimized

    def set_fuse_unk(self, fuse_unk: bool):
        self.fuse_unk = fuse_unk

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
    
class TestUnigramTrainer(unittest.TestCase):

    def test_unigram_chars(self):
        # Initialize the trainer
        trainer = UnigramTrainer(show_progress=False)#, seed_size=16)

        # Define sentences
        sentences = [
            ("This is a", 1),
            ("こんにちは友達", 1),
        ]

        # Get required characters
        required_chars = trainer.required_chars(sentences)
        self.assertEqual(len(required_chars), 13)

        # Generate seed sentence pieces
        table = trainer.make_seed_sentence_pieces(sentences, progress=None)

        # Expected results
        target_strings = [
            "s", "i", " ", "達", "友", "ん", "は", "に", "ち", "こ", "h", "a", "T", "is ", "s "
        ]
        
        # Group target strings by their scores
        score_to_strings = {
            -2.5649493574615367: {"s", "i", " "},           # freq = 2.0
            -3.258096538021482: {                           # freq = 1.0
                "達", "友", "ん", "は", "に", "ち", 
                "こ", "h", "a", "T"
            },
            -1.4663370687934272: {"is "},                   # freq = 6.0
            -1.8718021769015916: {"s "}                     # freq = 4.0
        }
        
        # Group actual results by score
        actual_score_to_strings = defaultdict(set)
        for string, score in table:
            actual_score_to_strings[round(score, 10)].add(string)

        # Compare the grouped results
        for expected_score, expected_strings in score_to_strings.items():
            rounded_score = round(expected_score, 10)
            actual_strings = actual_score_to_strings[rounded_score]
            
            # Debug print
            #print(f"\nComparing strings for score {rounded_score}:")
            #print(f"Expected: {expected_strings}")
            #print(f"Actual: {actual_strings}")
            
            self.assertEqual(
                expected_strings,
                actual_strings,
                f"Strings don't match for score {rounded_score}"
            )

        # Verify we didn't get any unexpected scores
        expected_scores = {round(score, 10) for score in score_to_strings.keys()}
        actual_scores = {round(score, 10) for score in actual_score_to_strings.keys()}
        self.assertEqual(
            expected_scores,
            actual_scores,
            "Got unexpected scores in results"
        )

    def test_initial_alphabet(self):
        # Initialize the trainer with an initial alphabet
        initial_alphabet = set(['a', 'b', 'c', 'd', 'e', 'f'])
        trainer = UnigramTrainer(show_progress=False, initial_alphabet=initial_alphabet)

        # Define the test sentences
        sentences = [("こんにちは友達", 1)]

        # Get the required characters
        required_chars = trainer.required_chars(sentences)

        # Define the expected set of required characters
        expected_chars = set(["こ", "ん", "に", "ち", "は", "友", "達", "a", "b", "c", "d", "e", "f"])

        # Assert that the required characters match the expected characters
        self.assertEqual(required_chars, expected_chars)

    def test_unk_token(self):
        # 1. Should add `unk_token` as the first special token
        trainer = UnigramTrainer(
            show_progress=False,
            special_tokens=[AddedToken("[SEP]", single_word=True), AddedToken("[CLS]", single_word=True)],
            unk_token="[UNK]"
        )
        unigram = Unigram()  # Default initialization

        # Run training
        trainer.do_train([("The", 12), ("are", 11)], unigram)

        # Check that `[UNK]` is the first special token
        pieces = unigram.get_vocab()
        self.assertEqual(list(pieces.keys())[:3], ["[UNK]", "[SEP]", "[CLS]"])

        # 2. Let `[UNK]` stay where it is in special tokens
        trainer = UnigramTrainer(
            show_progress=False,
            special_tokens=[
                AddedToken("[SEP]", single_word=True),
                AddedToken("[CLS]", single_word=True),
                AddedToken("[UNK]", single_word=True)
            ],
            unk_token="[UNK]"
        )
        unigram = Unigram()

        # Run training
        trainer.do_train([("The", 12), ("are", 11)], unigram)
        #print('unk_id:', unigram.unk_id)

        # Check that `[UNK]` appears in the original specified position
        pieces = unigram.get_vocab()
        self.assertEqual(list(pieces.keys())[:3], ["[SEP]", "[CLS]", "[UNK]"])

        ''' Ignoring this case for now; require UNK token to be in the vocab
        # 3. Don't add `[UNK]` if it's not specified as a required token
        trainer = UnigramTrainer(
            show_progress=False
        )
        unigram = Unigram()

        # Run training
        trainer.do_train([("The", 12), ("are", 11)], unigram)

        # Check that the first token is not `[UNK]`
        pieces = unigram.get_vocab()
        first_token = list(pieces.keys())[0]
        self.assertNotEqual(first_token, "[UNK]")  # Expecting another token, e.g., 'e'
       '''
    ''' # TODO: Ignore this case for now
    def test_special_tokens(self):
        # Initialize the trainer with `[SEP]` and `[CLS]` as special tokens
        trainer = UnigramTrainer(
            show_progress=False,
            special_tokens=[AddedToken("[SEP]", single_word=True), AddedToken("[CLS]", single_word=True)]
        )
        unigram = Unigram()  # Default initialization

        # Run training
        trainer.do_train([("The", 12), ("are", 11)], unigram)

        # Check that `[SEP]` and `[CLS]` are the first entries in the vocabulary
        pieces = unigram.get_vocab()
        self.assertEqual(list(pieces.keys())[:2], ["[SEP]", "[CLS]"])
    '''

class TestUnigram(unittest.TestCase):
    def test_populate_nodes_unk(self):
        # Initialize the model with a single `<unk>` token
        pieces = [("<unk>", 0.0)]
        model = Unigram(pieces, 0, False)

        # Create a lattice for the input "abc" with BOS and EOS IDs from the model
        lattice = Lattice("abc", model.bos_id, model.eos_id)
        model.populate_nodes(lattice)

        # Assertions to check lattice structure
        # Each position in "abc" should have a single `<unk>` node
        self.assertEqual(len(lattice.begin_nodes[0]), 1)
        self.assertEqual(len(lattice.begin_nodes[1]), 1)
        self.assertEqual(len(lattice.begin_nodes[2]), 1)

        # Check that the node IDs match the expected values
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[0][0]), 0)  # `<unk>` at position 0
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[1][0]), 0)  # `<unk>` at position 1
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[2][0]), 0)  # `<unk>` at position 2

        # Check that the node indices (node_id) are incrementing correctly
        self.assertEqual(lattice.begin_nodes[0][0], 2)
        self.assertEqual(lattice.begin_nodes[1][0], 3)
        self.assertEqual(lattice.begin_nodes[2][0], 4)

    def test_populate_nodes(self):
        # Initialize the model with vocabulary pieces
        pieces = [
            ("<unk>", 0.0),
            ("a", 0.1),
            ("b", 0.2),
            ("ab", 0.3),
            ("bc", 0.4),
        ]
        model = Unigram(pieces, 0, False)

        # Create a lattice for the input "abc" with BOS and EOS IDs from the model
        lattice = Lattice("abc", model.bos_id, model.eos_id)
        model.populate_nodes(lattice)

        # Assertions to check lattice structure
        # Position 0 (for 'a' and 'ab') should have 2 nodes, Position 1 (for 'b' and 'bc') should have 2 nodes
        # Position 2 should have 1 node (unknown for 'c')
        self.assertEqual(len(lattice.begin_nodes[0]), 2)
        self.assertEqual(len(lattice.begin_nodes[1]), 2)
        self.assertEqual(len(lattice.begin_nodes[2]), 1)

        # Use get_vocab_id to confirm each node's vocabulary ID
        # `id` refers to the vocabulary ID, `node_id` is the lattice's internal node index
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[0][0]), 1)  # 'a'
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[0][1]), 3)  # 'ab'
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[1][0]), 2)  # 'b'
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[1][1]), 4)  # 'bc'
        self.assertEqual(lattice.get_vocab_id(lattice.begin_nodes[2][0]), 0)  # '<unk>' for 'c'

        # Check that node indices (node_id) are assigned correctly
        self.assertEqual(lattice.begin_nodes[0][0], 2)  # Node ID for 'a' at position 0
        self.assertEqual(lattice.begin_nodes[0][1], 3)  # Node ID for 'ab' at position 0
        self.assertEqual(lattice.begin_nodes[1][0], 4)  # Node ID for 'b' at position 1
        self.assertEqual(lattice.begin_nodes[1][1], 5)  # Node ID for 'bc' at position 1
        self.assertEqual(lattice.begin_nodes[2][0], 6)  # Node ID for '<unk>' at position 2

    def test_encode(self):
        # Initialize the model with specific sentence pieces
        sentencepieces = [
            ("<unk>", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("c", 0.0),
            ("d", 0.0),
            ("cd", 1.0),
            ("ab", 2.0),
            ("abc", 5.0),
            ("abcd", 10.0),
        ]
        model = Unigram(sentencepieces, 0, False)

        # Encode the sentence "abcd" and check the result
        result = model.encode("abcd")
        self.assertEqual(result, ["abcd"])

    def test_encode2(self):
        # Initialize the model with specific sentence pieces
        sentencepieces = [
            ("<unk>", 0.0),
            ("ab", 0.0),
            ("cd", -0.1),
            ("abc", -0.2),
            ("a", -0.3),
            ("b", -0.4),
            ("c", -0.5),
            ("ABC", -0.5),
            ("abcdabcd", 20.0),
            ("q", 20.5),
            ("r", 20.5),
            ("qr", -0.5),
        ]
        model = Unigram(sentencepieces, 0, False)

        # TODO: implement optimized encoding
        for is_optimized in [True, False]:
            model.set_optimized(is_optimized)
            print(f"IsOptimized: {is_optimized}")
            self.assertEqual(model.encode("abc"), ["abc"])
            self.assertEqual(model.encode("AB"), ["AB"])

            model.set_fuse_unk(False)
            self.assertEqual(model.encode("AB"), ["A", "B"])
            model.set_fuse_unk(True)
            self.assertEqual(model.encode("AB"), ["AB"])

            self.assertEqual(model.encode("abcd"), ["ab", "cd"])
            self.assertEqual(model.encode("abcc"), ["abc", "c"])
            self.assertEqual(model.encode("xabcabaabcdd"), ["x", "abc", "ab", "a", "ab", "cd", "d"])

            model.set_fuse_unk(False)
            self.assertEqual(model.encode("xyz東京"), ["x", "y", "z", "東", "京"])
            model.set_fuse_unk(True)
            self.assertEqual(model.encode("xyz東京"), ["xyz東京"])

            # User encoded cases
            self.assertEqual(model.encode("ABC"), ["ABC"])
            self.assertEqual(model.encode("abABCcd"), ["ab", "ABC", "cd"])
            self.assertEqual(model.encode("ababcdabcdcd"), ["ab", "abcdabcd", "cd"])
            self.assertEqual(model.encode("abqrcd"), ["ab", "q", "r", "cd"])

    ''' # TODO: implement byte fallback
    def test_unigram_bytefallback(self):
        # Initialize the model with sentence pieces that include byte-level tokens
        sentencepieces = [
            ("<unk>", 0.0),
            ("<0xC3>", -0.01),
            ("<0xA9>", -0.03),
        ]
        model = Unigram(sentencepieces, 0, True)

        # Test tokenization for the character "é" (represented as bytes 0xC3 0xA9)
        tokens = model.tokenize("é")
        self.assertEqual(
            tokens,
            [
                Token(id=1, text="<0xC3>", offsets=(0, 2)),
                Token(id=2, text="<0xA9>", offsets=(0, 2)),
            ]
        )

        # Test tokenization for "?é" where "?" should trigger the unknown token
        tokens = model.tokenize("?é")
        self.assertEqual(tokens[0].id, 0)  # "<unk>" for "?"
    '''

if __name__ == "__main__":
    unittest.main()