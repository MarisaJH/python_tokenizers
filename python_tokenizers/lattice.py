import networkx as nx
import heapq
from dataclasses import dataclass, field
from typing import Optional, Any, List
import numpy as np
import unittest

@dataclass(order=True)
class Hypothesis:
    '''Hypothesis for n-best search'''
    fx: float # overall score
    node_ref: int = field(compare=False)  # node reference in the networkx graph
    next: Optional['Hypothesis'] = field(compare=False)
    gx: float = field(compare=False) # score so far

    def __init__(self, node_ref, next_hypo, fx, gx):
        self.node_ref = node_ref
        self.next = next_hypo
        self.fx = -fx  # negate for max-heap behavior in heapq
        self.gx = gx

class Lattice:
    def __init__(self, sentence, bos_id, eos_id):
        self.sentence_ = sentence
        self.len_ = len(sentence)
        self.graph = nx.DiGraph()  # Directed graph for lattice
        self.node_count = 0  # To assign unique node IDs
        
        # handle BOS and EOS
        self._bos_id = bos_id # vocab id
        self._eos_id = eos_id
        self.begin_nodes = [[] for _ in range(self.len_ + 1)] # holds unique node ids (NOT vocab ids)
        self.end_nodes = [[] for _ in range(self.len_ + 1)]       

        bos_node_id = 0
        eos_node_id = 1
        self.graph.add_node(bos_node_id, id=bos_id, pos=0, length=0, score=0.0, backtrace_score=0.0, prev=None)
        self.graph.add_node(eos_node_id, id=eos_id, pos=self.len_, length=0, score=0.0, backtrace_score=0.0, prev=None)
        self.node_count += 2  # increment node count for BOS and EOS nodes
        
        # Add BOS to end_nodes[0] and EOS to begin_nodes[len]
        self.end_nodes[0].append(bos_node_id)
        self.begin_nodes[self.len_].append(eos_node_id)
        
    def __str__(self):
        """Display function to mimic the Rust Lattice fmt."""
        def display_pieces(nodes):
            return [
                [self.piece(node_id) for node_id in node_list]  # Changed from self.graph.nodes[node_id]['node']
                for node_list in nodes
            ]

        return f"Lattice(sentence='{self.sentence_}', begin_nodes={display_pieces(self.begin_nodes)}, end_nodes={display_pieces(self.end_nodes)})"
    
    def log_sum_exp(self, x, y, init_mode=False):
        if init_mode:
            return y
        else:
            '''
            vmin, vmax = (y, x) if x > y else (x, y)
            k_minus_log_epsilon = 50.0
            # If the difference is too large, return vmax directly
            if vmax > vmin + k_minus_log_epsilon:
                return vmax
            else:
                # Use numpy's logaddexp for the main calculation
                return np.logaddexp(vmin, vmax)

            '''
               # Debug prints
            #print(f"  log_sum_exp: x={x}, y={y}, init_mode={init_mode}")
            
            vmin, vmax = (y, x) if x > y else (x, y)
            #print(f"  vmin={vmin}, vmax={vmax}")
            
            k_minus_log_epsilon = 50.0
            if vmax > vmin + k_minus_log_epsilon:
                #print("  returning vmax due to large difference")
                return vmax
                
            else:
                result = np.logaddexp(vmin, vmax)
                #print(f"  result={result}")
                return result

    def insert(self, pos, length, score, id):
        node_id = self.node_count
        self.node_count += 1  # increment node count to keep node IDs unique

        self.graph.add_node(node_id, # unique to each node
                            id=id, # vocab id of the token (may not be unique)
                            pos=pos,
                            length=length,
                            score=score,
                            backtrace_score=0.0,
                            prev=None)

        # track node's position in begin_nodes and end_nodes
        self.begin_nodes[pos].append(node_id)
        end_pos = pos + length
        if end_pos <= self.len_:
            self.end_nodes[end_pos].append(node_id)

    def viterbi(self, return_vocab_ids=False) -> list:
        # forward pass to compute backtrace scores and previous nodes
        for pos in range(self.len_ + 1):
            if not self.begin_nodes[pos]:
                return []
                
            for rnode_id in self.begin_nodes[pos]:
                
                rnode = self.graph.nodes[rnode_id]
                rnode['prev'] = None  

                best_score = -1e10#float("-inf")
                best_node_id = None

                for lnode_id in self.end_nodes[pos]:
                    lnode = self.graph.nodes[lnode_id]
                    score = lnode['backtrace_score'] + rnode['score']
                    if best_node_id is None or score > best_score:
                        best_node_id = lnode_id  
                        best_score = score

                if best_node_id is not None:
                    # Update rnode with the best scoring node and its score
                    rnode['prev'] = best_node_id
                    rnode['backtrace_score'] = best_score
                else:
                    return []  # Handle case where no valid best node is found
                    
        # Backtrack for best path
        results = []
        node_id = self.begin_nodes[self.len_][0]  
        node = self.graph.nodes[node_id] 

        while node.get('prev') is not None:
              
            node_id = node['prev']  
            node = self.graph.nodes[node_id]  

            # break if bos node is reached
            if node_id == self.bos_node():
                break

            results.append(node_id)

        results.reverse()
        if return_vocab_ids:
            return [self.graph.nodes[node_id]['id'] for node_id in results]
        
        return results
    
    def piece(self, node_id):
        """retrieve substring of sentence corresponding to a given node ID"""
        node_data = self.graph.nodes[node_id]  
        pos = node_data['pos']
        length = node_data['length']
        return self.sentence_[pos:pos + length]

    def tokens(self):
        """call viterbi to find best sequence of node IDs and return corresponding tokens"""
        best_path_ids = self.viterbi()  # node IDs from viterbi
        return [self.piece(node_id) for node_id in best_path_ids]

    def nbest(self, n: int, return_vocab_ids=False) -> List[List[int]]:
        if n == 0:
            return []
        elif n == 1:
            return [self.viterbi()]
        
        agenda = [] # pqueue of Hypothesis
        hypotheses = [] # list of hypotheses, each a list of node IDs
        
        eos = self.eos_node()
        eos_node = self.graph.nodes[eos]
        hypo = Hypothesis(eos, None, eos_node['score'], eos_node['score'])
        heapq.heappush(agenda, hypo)

        # Fill backtrace scores with viterbi
        self.viterbi()
        
        while agenda:
            top = heapq.heappop(agenda)
            node_id = top.node_ref
            node = self.graph.nodes[node_id]
            if node_id == self.bos_node():
                # Collect the hypothesis
                hypothesis = []
                next_hypo = top.next
                while next_hypo.node_ref != self.eos_node():
                    hypothesis.append(next_hypo.node_ref)
                    next_hypo = next_hypo.next
                hypotheses.append(hypothesis)
                
                if len(hypotheses) == n:
                    return hypotheses
            else:
                for lnode_id in self.end_nodes[node['pos']]:
                    lnode = self.graph.nodes[lnode_id]
                    top_gx = top.gx
                    fx = lnode['backtrace_score'] + top_gx
                    gx = lnode['score'] + top_gx
                    new_hypo = Hypothesis(lnode_id, top, fx, gx)
                    heapq.heappush(agenda, new_hypo)
                
                # Dynamic agenda shrinking
                k_max_agenda_size = 100_000
                k_min_agenda_size = 512
                if len(agenda) > k_max_agenda_size:
                    # Reduce the size of the agenda to k_min_agenda_size or 10x `n`
                    agenda = heapq.nlargest(min(k_min_agenda_size, n * 10), agenda)
                    heapq.heapify(agenda)
        if return_vocab_ids:
            return [[self.graph.nodes[node_id]['id'] for node_id in path] for path in hypotheses]
        return hypotheses

    def nbest_tokens(self, n: int) -> List[List[str]]:
        return [
            [self.piece(node) for node in path] # node ids to tokens
            for path in self.nbest(n)
        ]
    
    def len(self) -> int:
        return self.len_

    def is_empty(self) -> bool:
        return self.len_ == 0
    
    def bos_node(self) -> int:
        """Return the ID of the begin-of-sequence node."""
        return self.end_nodes[0][0]

    def eos_node(self) -> int:
        """Return the ID of the end-of-sequence node."""
        return self.begin_nodes[self.len_][0]
    
    def get_vocab_id(self, node_id: int) -> int:
        """
        Retrieve the vocab ID associated with a given node ID.
        
        Args:
        - `node_id`: The ID of the node in the lattice.
        
        Returns:
        - The vocab ID associated with this node.
        """
        return self.graph.nodes[node_id]['id'] 
    
    def sentence(self) -> str:
        return self.sentence_
    
    def surface(self, n: int) -> str:
        """Return the substring of the sentence starting from the nth character."""
        return self.sentence_[n:] if n < len(self.sentence_) else ""
    
    def populate_marginal(self, freq: float, expected: np.ndarray) -> float:
        len_ = self.len_
        n_nodes = self.node_count
        
        alpha = np.zeros(n_nodes)
        beta = np.zeros(n_nodes)
        
        # Forward pass (alpha values)
        #print("\nForward pass:")
        for pos in range(len_ + 1):
            #print(f"\nPosition {pos}")
            for rnode_id in self.begin_nodes[pos]:
                rnode = self.graph.nodes[rnode_id]
                #print(f"Right node {rnode_id}: token={self.piece(rnode_id)}, score={rnode['score']}")
                
                for lnode_id in self.end_nodes[pos]:
                    lnode = self.graph.nodes[lnode_id]
                    new_score = lnode['score'] + alpha[lnode_id]
                    #print(f"  Left node {lnode_id}: token={self.piece(lnode_id)}")
                    #print(f"    lnode score: {lnode['score']}")
                    #print(f"    alpha[lnode_id]: {alpha[lnode_id]}")
                    #print(f"    new_score: {new_score}")
                    
                    alpha[rnode_id] = self.log_sum_exp(
                        alpha[rnode_id],
                        new_score,
                        init_mode=(lnode_id == self.end_nodes[pos][0])
                    )
                    #print(f"    alpha[rnode_id] after update: {alpha[rnode_id]}")
        
        #print("\nBackward pass:")
        # Backward pass (beta values)
        for pos in range(len_, -1, -1):
            #print(f"\nPosition {pos}")
            for lnode_id in self.end_nodes[pos]:
                lnode = self.graph.nodes[lnode_id]
                #print(f"Left node {lnode_id}: token={self.piece(lnode_id)}, score={lnode['score']}")
                
                for rnode_id in self.begin_nodes[pos]:
                    rnode = self.graph.nodes[rnode_id]
                    new_score = rnode['score'] + beta[rnode_id]
                    #print(f"  Right node {rnode_id}: token={self.piece(rnode_id)}")
                    #print(f"    rnode score: {rnode['score']}")
                    #print(f"    beta[rnode_id]: {beta[rnode_id]}")
                    #print(f"    new_score: {new_score}")
                    
                    beta[lnode_id] = self.log_sum_exp(
                        beta[lnode_id],
                        new_score,
                        init_mode=(rnode_id == self.begin_nodes[pos][0])
                    )
                    #print(f"    beta[lnode_id] after update: {beta[lnode_id]}")

        # Calculate normalization factor z
        eos_id = self.eos_node()
        z = alpha[eos_id]

        # Update expected counts
        for pos in range(len_):
            for node_id in self.begin_nodes[pos]:
                node = self.graph.nodes[node_id]
                vocab_id = node['id']  # Vocabulary ID
                a = alpha[node_id]
                b = beta[node_id]
                total = a + node['score'] + b - z
                update = freq * np.exp(total)
                expected[vocab_id] += update

        return freq * z

    def sample(self, theta: float) -> List[int]:
        len_ = self.len_
        if len_ == 0:
            return []

        n_nodes = self.node_count
        alpha = np.zeros(n_nodes)

        # Forward pass (compute alpha values)
        for pos in range(len_ + 1):
            for rnode_id in self.begin_nodes[pos]:
                for lnode_id in self.end_nodes[pos]:
                    lnode = self.graph.nodes[lnode_id]
                    rnode = self.graph.nodes[rnode_id]

                    alpha[rnode_id] = np.logaddexp(
                        alpha[rnode_id],
                        theta * (lnode['score'] + alpha[lnode_id]),
                        where=lnode_id == self.end_nodes[pos][0]
                    )

        # Sampling backtrack
        results = []
        eos_id = self.eos_node()
        z = alpha[eos_id]
        node_id = eos_id

        while True:
            pos = self.graph.nodes[node_id]['pos']
            probs = np.array(len(self.end_nodes[pos]), dtype=np.float64)

            # Calculate probabilities for transitions to each node in end_nodes at 'pos'
            for i, lnode_id in enumerate(self.end_nodes[pos]):
                lnode = self.graph.nodes[lnode_id]
                prob = np.exp(alpha[lnode_id] + theta * lnode['score'] - z)
                #probs.append(prob)
                probs[i] = prob

            # Sample from the weighted distribution
            probs /= probs.sum()  # normalize probabilities
            sampled_index = np.random.choice(len(probs), p=probs)
            node_id = self.end_nodes[pos][sampled_index]

            # Break if we reach the BOS node
            if node_id == self.bos_node():
                break

            # Update 'z' and add 'node_id' to results
            z = alpha[node_id]
            results.append(node_id)

        results.reverse()
        return results

    def sample_token(self, theta: float) -> List[str]:
        """Generate a sampled token sequence based on the temperature parameter theta."""
        sampled_node_ids = self.sample(theta)
        
        # convert node IDs to corresponding tokens
        return [self.piece(node_id) for node_id in sampled_node_ids]


class TestLattice(unittest.TestCase):
    def test_set_sentence(self):
        # Initialize an empty sentence with BOS and EOS vocab IDs 1 and 2, respectively
        lattice = Lattice("", 1, 2)

        # Test length and sentence properties
        self.assertEqual(lattice.len(), 0)
        self.assertEqual(lattice.sentence(), "")
        self.assertEqual(lattice.surface(0), "")

        # Initialize a non-empty sentence
        lattice = Lattice("test", 1, 2)
        self.assertEqual(lattice.len(), 4)
        self.assertEqual(lattice.sentence(), "test")
        self.assertEqual(lattice.surface(0), "test")
        self.assertEqual(lattice.surface(1), "est")
        self.assertEqual(lattice.surface(2), "st")
        self.assertEqual(lattice.surface(3), "t")

        # Check BOS and EOS node IDs and vocab IDs
        bos_id = lattice.bos_node()  # Node ID for BOS, should be 0
        eos_id = lattice.eos_node()  # Node ID for EOS, should be the last node ID added

        # Check that BOS and EOS have the expected node IDs
        self.assertEqual(bos_id, 0)
        self.assertEqual(eos_id, 1)
        self.assertEqual(lattice.graph.nodes[bos_id]['id'], 1)  # Vocab ID for BOS
        self.assertEqual(lattice.graph.nodes[eos_id]['id'], 2)  # Vocab ID for EOS
        self.assertEqual(lattice.end_nodes[0][0], bos_id)
        self.assertEqual(lattice.begin_nodes[4][0], eos_id)

        # Test with a mixed Japanese and English sentence
        lattice = Lattice("テストab", 1, 2)
        self.assertEqual(lattice.len(), 5)  # Length in characters
        self.assertEqual(lattice.sentence(), "テストab")
        self.assertEqual(lattice.surface(0), "テストab")
        self.assertEqual(lattice.surface(1), "ストab")
        self.assertEqual(lattice.surface(2), "トab")
        self.assertEqual(lattice.surface(3), "ab")
        self.assertEqual(lattice.surface(4), "b")

    def test_insert(self):
        # Initialize the lattice with a mixed sentence "ABあい" and BOS/EOS IDs
        lattice = Lattice("ABあい", 1, 2)

        # Insert nodes with different character positions, lengths, and IDs
        lattice.insert(0, 1, 0.0, 3)  # "A"
        lattice.insert(1, 1, 0.0, 4)  # "B"
        lattice.insert(2, 1, 0.0, 5)  # "あ"
        lattice.insert(3, 1, 0.0, 6)  # "い"
        lattice.insert(0, 2, 0.0, 7)  # "AB"
        lattice.insert(1, 2, 0.0, 8)  # "Bあ"
        lattice.insert(2, 2, 0.0, 9)  # "あい"

        # Access the nodes based on insertion order (skipping BOS and EOS)
        node0 = lattice.graph.nodes[2]  # Node ID 2 (first non-BOS/EOS)
        node1 = lattice.graph.nodes[3]
        node2 = lattice.graph.nodes[4]
        node3 = lattice.graph.nodes[5]
        node4 = lattice.graph.nodes[6]
        node5 = lattice.graph.nodes[7]
        node6 = lattice.graph.nodes[8]

        # Verify the tokens represented by each node
        self.assertEqual(lattice.piece(2), "A")
        self.assertEqual(lattice.piece(3), "B")
        self.assertEqual(lattice.piece(4), "あ")
        self.assertEqual(lattice.piece(5), "い")
        self.assertEqual(lattice.piece(6), "AB")
        self.assertEqual(lattice.piece(7), "Bあ")
        self.assertEqual(lattice.piece(8), "あい")

        # Verify positions in character-based indexing
        self.assertEqual(node0['pos'], 0)
        self.assertEqual(node1['pos'], 1)
        self.assertEqual(node2['pos'], 2)
        self.assertEqual(node3['pos'], 3)
        self.assertEqual(node4['pos'], 0)
        self.assertEqual(node5['pos'], 1)
        self.assertEqual(node6['pos'], 2)

        # Verify lengths in character-based indexing
        self.assertEqual(node0['length'], 1)
        self.assertEqual(node1['length'], 1)
        self.assertEqual(node2['length'], 1)
        self.assertEqual(node3['length'], 1)
        self.assertEqual(node4['length'], 2)
        self.assertEqual(node5['length'], 2)
        self.assertEqual(node6['length'], 2)

        # Verify BOS and EOS IDs
        bos_id = lattice.bos_node()
        eos_id = lattice.eos_node()
        self.assertEqual(lattice.graph.nodes[bos_id]['id'], 1)
        self.assertEqual(lattice.graph.nodes[eos_id]['id'], 2)

        # Verify the vocabulary IDs assigned to each inserted node
        self.assertEqual(node0['id'], 3)
        self.assertEqual(node1['id'], 4)
        self.assertEqual(node2['id'], 5)
        self.assertEqual(node3['id'], 6)
        self.assertEqual(node4['id'], 7)
        self.assertEqual(node5['id'], 8)
        self.assertEqual(node6['id'], 9)

        # Check the lengths of begin_nodes and end_nodes at specific positions
        self.assertEqual(len(lattice.begin_nodes[0]), 2)
        self.assertEqual(len(lattice.begin_nodes[1]), 2)
        self.assertEqual(len(lattice.begin_nodes[2]), 2)
        self.assertEqual(len(lattice.begin_nodes[3]), 1)
        self.assertEqual(len(lattice.begin_nodes[4]), 1)

        self.assertEqual(len(lattice.end_nodes[0]), 1)
        self.assertEqual(len(lattice.end_nodes[1]), 1)
        self.assertEqual(len(lattice.end_nodes[2]), 2)
        self.assertEqual(len(lattice.end_nodes[3]), 2)
        self.assertEqual(len(lattice.end_nodes[4]), 2)

        # Verify node IDs in begin_nodes and end_nodes lists
        self.assertEqual(lattice.begin_nodes[0][0], 2)
        self.assertEqual(lattice.begin_nodes[0][1], 6)
        self.assertEqual(lattice.begin_nodes[1][0], 3)
        self.assertEqual(lattice.begin_nodes[1][1], 7)
        self.assertEqual(lattice.begin_nodes[2][0], 4)
        self.assertEqual(lattice.begin_nodes[2][1], 8)
        self.assertEqual(lattice.begin_nodes[3][0], 5)
        self.assertEqual(lattice.begin_nodes[4][0], eos_id)

        self.assertEqual(lattice.end_nodes[0][0], bos_id)
        self.assertEqual(lattice.end_nodes[1][0], 2)
        self.assertEqual(lattice.end_nodes[2][0], 3)
        self.assertEqual(lattice.end_nodes[2][1], 6)
        self.assertEqual(lattice.end_nodes[3][0], 4)
        self.assertEqual(lattice.end_nodes[3][1], 7)
        self.assertEqual(lattice.end_nodes[4][0], 5)
        self.assertEqual(lattice.end_nodes[4][1], 8)

    def test_viterbi(self):
        # Initialize the lattice with the sentence "ABC" and BOS/EOS IDs
        lattice = Lattice("ABC", 1, 2)

        # Initially, the lattice is incomplete, so viterbi() should return an empty list
        self.assertEqual(lattice.viterbi(), [])

        # Insert a node at position 0 with length 1 and ID 3
        lattice.insert(0, 1, 0.0, 3)
        # The lattice is still incomplete, so viterbi() should return an empty list
        self.assertEqual(lattice.viterbi(), [])

        # Insert additional nodes to complete the lattice path
        lattice.insert(1, 1, 0.0, 4)
        lattice.insert(2, 1, 0.0, 5)

        # Now, viterbi() should return a path with a length of 3
        self.assertEqual(len(lattice.viterbi()), 3)

    def test_viterbi2(self):
        # Initialize the lattice with the sentence "ABC" and BOS/EOS IDs
        lattice = Lattice("ABC", 1, 2)

        # Insert nodes with specific positions, lengths, and scores
        lattice.insert(0, 1, 0.0, 3)  # "A"
        lattice.insert(1, 1, 0.0, 4)  # "B"
        lattice.insert(2, 1, 0.0, 5)  # "C"

        # Check that tokens() returns ["A", "B", "C"] as the optimal path
        self.assertEqual(lattice.tokens(), ["A", "B", "C"])

        # Insert a node spanning "AB" with a higher score to form ["AB", "C"]
        lattice.insert(0, 2, 2.0, 6)
        self.assertEqual(lattice.tokens(), ["AB", "C"])

        # Insert a node spanning "BC" with an even higher score to form ["A", "BC"]
        lattice.insert(1, 2, 5.0, 7)
        self.assertEqual(lattice.tokens(), ["A", "BC"])

        # Insert a node spanning "ABC" with the highest score to form ["ABC"]
        lattice.insert(0, 3, 10.0, 8)
        self.assertEqual(lattice.tokens(), ["ABC"])

    def test_nbest(self):
        # Initialize the lattice with the sentence "ABC" and BOS/EOS IDs
        lattice = Lattice("ABC", 1, 2)

        # Insert nodes with specific positions, lengths, and scores
        lattice.insert(0, 1, 0.0, 3)  # "A"
        lattice.insert(1, 1, 0.0, 4)  # "B"
        lattice.insert(2, 1, 0.0, 5)  # "C"
        lattice.insert(0, 2, 2.0, 6)  # "AB"
        lattice.insert(1, 2, 5.0, 7)  # "BC"
        lattice.insert(0, 3, 10.0, 8)  # "ABC"

        # Retrieve the top 10 best paths
        nbests = lattice.nbest_tokens(10)

        # Expected paths in order of their scores
        expected_nbests = [
            ["ABC"],
            ["A", "BC"],
            ["AB", "C"],
            ["A", "B", "C"]
        ]

        # Check if nbests matches the expected paths
        self.assertEqual(nbests, expected_nbests)

        # Test edge cases with different values of n
        self.assertEqual(lattice.nbest_tokens(0), [])  # Should be empty for n=0
        self.assertEqual(lattice.nbest_tokens(1), [["ABC"]])  # Top 1 best path should be ["ABC"]

    def test_populate(self):
        # Initialize lattice with the sentence "ABC" and BOS/EOS IDs
        lattice = Lattice("ABC", 1, 2)

        # Insert nodes with specific positions, lengths, and scores
        lattice.insert(0, 1, 1.0, 3)  # "A"
        lattice.insert(1, 1, 1.2, 4)  # "B"
        lattice.insert(2, 1, 2.5, 5)  # "C"
        lattice.insert(0, 2, 3.0, 6)  # "AB"
        lattice.insert(1, 2, 4.0, 7)  # "BC"
        lattice.insert(0, 3, 2.0, 8)  # "ABC"

        # Expected partition function components
        p1 = np.exp(1.0 + 1.2 + 2.5)
        p2 = np.exp(3.0 + 2.5)
        p3 = np.exp(1.0 + 4.0)
        p4 = np.exp(2.0)
        z = p1 + p2 + p3 + p4

        # Initialize probabilities array
        probs = np.zeros(9)
        #print('probs:', probs)

        # Calculate log partition function using populate_marginal
        log_z = lattice.populate_marginal(1.0, probs)
        #print('probs:', probs)

        # Assert that log partition function (log_z) is close to log(z)
        self.assertAlmostEqual(log_z, np.log(z), places=3)

        # Expected marginal probabilities for each node
        expected_probs = [0.0, 0.0, 0.0, (p1 + p3) / z, p1 / z, (p1 + p2) / z,
                          p2 / z, p3 / z, p4 / z]

        # Check that each probability matches the expected value
        for i, expected_prob in enumerate(expected_probs):
            self.assertAlmostEqual(probs[i], expected_prob, places=3)

if __name__ == '__main__':
    unittest.main()