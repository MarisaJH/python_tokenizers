from tokenizers import normalizers, Regex, pre_tokenizers, decoders
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional
from .unigram import Unigram, UnigramTrainer
import os, pickle, json

from cltk.sentence.lat import LatinPunktSentenceTokenizer


class UnigramTokenizer(PreTrainedTokenizer):
    def __init__(self, model: Unigram, *args, **kwargs):
        """
        Args:
            model: the Unigram model object

        """
        self.model = model
        
        self.normalizer = normalizers.Sequence([
            normalizers.NFD(), 
            normalizers.Lowercase(), 
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " ")
        ])
        #self.pre_tokenizer = BasicTokenizer()
        self.pre_tokenizer = pre_tokenizers.Metaspace()
        self.decoder = decoders.Metaspace()
        
        super().__init__(*args, **kwargs)
        
        self.cls_id = self.model.token_to_id(self.cls_token)
        self.sep_id = self.model.token_to_id(self.sep_token)
        '''
        self.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.cls_id),
                ("[SEP]", self.sep_id),
            ],
        )
        '''

        self.unk_id = self.model.unk_id
        #self.unk_token = self.model.id_to_token(self.unk_id) 
        
        #self.special_tokens_map = {}
        #for key, value in kwargs.items():
        #    if key in self.SPECIAL_TOKENS_ATTRIBUTES:
        #        self.special_tokens_map[key] = value
        

    def vocab_size(self) -> int:
        # size without added tokens
        return len(self.model)

    def _convert_token_to_id(self, token: str) -> int:
        #print(token, self.model.token_to_id(token))
        id_ = self.model.token_to_id(token)
        if id_ is None:
            return self.unk_id
        return id_

    
    def _convert_id_to_token(self, id: int) -> str:
        token = self.model.id_to_token(id)
        if token is None:
            return self.unk_token
        return token
    
    def _tokenize(self, text: str) -> List[str]:
        normalized_text = self.normalizer.normalize_str(text)
        pre_tokenized = self.pre_tokenizer.pre_tokenize_str(normalized_text)
        # if pre_tokenized is a list of tuples, extract the first element
        if isinstance(pre_tokenized[0], tuple):
            pre_tokenized = [t[0] for t in pre_tokenized]

        encoded = []
        #print('in UnigramTokenizer._tokenize')
        for word in pre_tokenized:
            ids = self.model.encode(word)
            #print(word, ids)
            encoded.extend(ids)
        #return [token for word in pre_tokenized for token in self.model.encode(word)]
        return encoded
    
    def get_vocab(self) -> Dict[str, int]:
        return self.model.get_vocab()
    
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # Convert token IDs to tokens
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        #print('Filtered tokens:', filtered_tokens)

        sub_texts = []
        current_sub_text = []

        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            '''
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
            '''
            #current_sub_text.append(token)
            sub_texts.append(token)

        #if current_sub_text:
        #    sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # Use the decoder if available
        if self.decoder:
            return self.decoder.decode(sub_texts)
        else:
            # Join sub_texts with or without spaces between special tokens
            if spaces_between_special_tokens:
                text = " ".join(sub_texts)
            else:
                text = "".join(sub_texts)

            # Clean up tokenization spaces if needed
            clean_up_tokenization_spaces = (
                clean_up_tokenization_spaces
                if clean_up_tokenization_spaces is not None
                else self.clean_up_tokenization_spaces
            )
            if clean_up_tokenization_spaces:
                text = self.clean_up_tokenization(text)

            return text
        
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_id] + token_ids_0 + [self.sep_id]
        return [self.cls_id] + token_ids_0 + [self.sep_id] + token_ids_1 + [self.sep_id]

    def save_pretrained(self, save_directory: str):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.model.save(save_directory)

        if save_directory.endswith('/'):
            save_directory = save_directory[:-1]
        with open(save_directory + "/model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # save special tokens map
        with open(save_directory + "/special_tokens_map.json", "w") as f:
            json.dump(self.special_tokens_map, f)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *init_inputs, **kwargs):
        # is path to model.pkl file, or directory containing model.pkl?
        if pretrained_model_name_or_path.endswith('.pkl'):
            model = Unigram.from_file(pretrained_model_name_or_path)
        else:
            if pretrained_model_name_or_path.endswith('/'):
                pretrained_model_name_or_path = pretrained_model_name_or_path[:-1]
            model = Unigram.from_file(pretrained_model_name_or_path + "/model.pkl")

        # load special tokens map
        with open(pretrained_model_name_or_path + "/special_tokens_map.json", "r") as f:
            special_tokens_map = json.load(f)

        # pass keys and values of special_tokens_map as kwargs to the constructor
        return cls(model=model, *init_inputs, **special_tokens_map)
    
    def train_from_iterator(self, iterator, trainer: Optional[UnigramTrainer] = None, N: int = None):
        #total_length = sum(len(batch) for batch in iterator)
        #max_read = 1_000_000

        progress = None
        if trainer and trainer.should_show_progress():
            from tqdm import tqdm
            progress = tqdm(total=N, desc="Pre-processing batches")

        words_count: Dict[str, int] = {}
        for batch in iterator:
            for sequence in batch:
                #print('sequence:', sequence)
                if progress:
                    progress.update(len(sequence))

                normalized = self.normalizer.normalize_str(sequence)
                #print('normalized:', normalized)
                pre_tokenized = self.pre_tokenizer.pre_tokenize_str(normalized)
                #print('pre_tokenized:', pre_tokenized)
                # Assuming pre_tokenized is a list of tuples, extract the first element
                if isinstance(pre_tokenized[0], tuple):
                    pre_tokenized = [t[0] for t in pre_tokenized]

                # Update the words_count dictionary
                for word in pre_tokenized:
                    words_count[word] = words_count.get(word, 0) + 1

        if progress:
            progress.close()

        trainer.words = words_count
        #print('words_count length:', len(words_count))

        special_tokens = trainer.train(self.model)
        #print('special_tokens:', special_tokens)
        #print('map:', self.special_tokens_map)
        inv_special_tokens = {v: k for k, v in self.special_tokens_map.items()}
        special_tokens_dict = {} # {str: AddedToken}
        for token in special_tokens:
            if token.content in inv_special_tokens:
                special_tokens_dict[inv_special_tokens[token.content]] = token 

        self.add_special_tokens(special_tokens_dict)  
    

if __name__ == "__main__":
    SAVE_DIR = '/work/pi_brenocon_umass_edu/marisa/latin-bert-morph/tokenizers/'
    tokenizer_type = 'Unigram'
    train_set = 'non-ia-subsampled'
    vocab_size = 30000
    tokenizer_save_dir= SAVE_DIR + f"{tokenizer_type}-tokenizer_train-{train_set}_v-{vocab_size}"
    model_file = tokenizer_save_dir + "/model.pkl"

    tokenizer = UnigramTokenizer.from_pretrained(model_file)
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_dir)
    print(tokenizer.vocab_size())

    sent_splitter = LatinPunktSentenceTokenizer()
    text = "Si autem donum universaliter sumatur pro omni eo quod ex Dei munere possidemus, sic caritas simpliciter potissima est, quae nos fini conjungit. [ 1345 ] Super Sent . , lib . 1 d. 17 q. 2 pr . Hic quaeritur , si caritas spiritus sanctus est , cum ipsa augeatur et minuatur in homine , utrum concedendum sit quod spiritus sanctus augeatur vel minuatur in homine . In parte ista Magister objicit contra praedeterminata , et dividitur in partes duas : in prima objicit per rationem ; in secunda per auctoritates , ibi : supra dictum est , quod spiritus sanctus caritas est patris et filii . Prima in tres : primo ponit objectionem ; secundo responsionem , ibi : his itaque respondemus dicentes , quod spiritus sanctus , sive caritas , penitus immutabilis est ; tertio responsionis confirmationem , ibi : ut autem certius fiat quod diximus , auctoritate confirmamus . Objectio autem est duplex , ut patet in littera , et similiter solutio . Supra dictum est , quod spiritus sanctus caritas est patris et filii . Hic objicit per auctoritates , et dividitur in tres partes , secundum tria , quae contra Magistrum ex auctoritatibus eliciuntur . Primum est quod alia est dilectio qua nos diligimus Deum , et Deus diligit nos . Dilectio autem qua Deus diligit nos , est spiritus sanctus . Ergo dilectio qua nos diligimus Deum , est aliud quam spiritus sanctus . Secundum est , quod dicitur caritas esse ex Deo , sicut et fides . Sed fides est ex Deo , ita quod non est Deus . Ergo et caritas ; et hoc ponit ibi : sed aliud est , inquiunt , quod magis urget . Tertium est quod caritas dicitur affectio vel motus mentis . Sed spiritus sanctus non est hujusmodi . Ergo etc . , et hoc ponit ibi : alias quoque inducunt rationes ad idem ostendendum . Quaelibet harum partium dividitur in objectionem et solutionem . Ad intelligentiam hujus partis quinque quaeruntur : 1 utrum caritas augeatur ; 2 de modo augmenti ; 3 utrum quolibet actu augeatur ; 4 utrum sit aliquis terminus augmenti ; 5 utrum diminuatur . [ 1347 ] Super Sent . , lib . 1 d. 17 q. 2 a. 1 arg . 1 Ad primum sic proceditur . Videtur quod caritas non augeatur . Nihil enim augetur nisi quantum . Sed nullum simplex est quantum , quia omne quantum est divisibile . Caritas autem est simplex habitus , et ita non est quantum per se , nec similiter per accidens , cum ejus subjectum , scilicet anima , sit etiam indivisibile . Ergo non augetur . "
    sents = sent_splitter.tokenize(text)

    sent = sents[0]
    print('Sentence:', sent)
    encoded = tokenizer.encode(sents[0], sents[1])
    print('Encoded:', encoded)
    print('Subwords:', [tokenizer._convert_id_to_token(id) for id in encoded])
    decoded = tokenizer.decode(encoded)
    print('Decoded:', decoded)

    #print("Test save_pretrained")
    #tokenizer.save_pretrained(tokenizer_save_dir + "_test")

