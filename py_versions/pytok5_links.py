import json
import time
import heapq
from collections import Counter
from dataclasses import dataclass, field

@dataclass 
class Node:
    next_: None
    prev_: None
    seq: list = field(default_factory=list)
    val: int = 0
    def __hash__(self):
        # Allows the node to be a key in a dictionary/counter
        return hash(self.val)
    
    def __eq__(self, other):
        # compares this Node to another node by checking the value
        if not isinstance(other, Node):
            return False
        return self.val == other.val
    def __repr__(self):
        return f"next: {self.next_} prev: {self.prev_} val:{self.val} "

@dataclass 
class chain:
    def __init__(self, toks):
        self.nodes = None
        self.num_nodes = 0
        self.symbols = 256
        self.top_pair = [-1,-1]

        counts = Counter()
        self.nodes = Node(next_= None, prev_=None) # Sentinel Head
        curr = self.nodes
        for val in toks:
            new_node = Node(next_=None, prev_=curr, val=val, seq=[val])
            if curr is not self.nodes: # If we have a previous real node
                pair = (curr.val, new_node.val)
                counts[pair] += 1 # Counter handles the "if in set" check for you
                
            curr.next_ = new_node
            curr = new_node
            self.num_nodes += 1
        
        self.persistent_counts = counts
        # thats everything on the counts
        counts = sorted(counts.items(), key=lambda x: -x[1])
        
        # priority queue
        self.heapq_ = [(-c, p) for p, c in self.persistent_counts.items()]
        heapq.heapify(self.heapq_)


    def Update(self):
        top_pair, freq = self.find_best()
        if not top_pair: return   
        
        self.symbols += 1
        curr = self.nodes.next_
        
        while curr is not None and curr.next_ is not None:
            if (curr.val, curr.next_.val) == top_pair:
                # --- SURGICAL UPDATE ---
                prev_node = curr.prev_
                target = curr.next_  # This will become our merged node
                next_node = target.next_
                
                # A. Remove dead pairs from counter
                # Pair to the left: (Prev, Curr)
                if prev_node and prev_node is not self.nodes:
                    self.persistent_counts[(prev_node.val, curr.val)] -= 1
                
                # Pair to the right: (Target, Next)
                if next_node:
                    self.persistent_counts[(target.val, next_node.val)] -= 1
                
                # B. Perform the Merge
                target.val = self.symbols
                target.seq = curr.seq + target.seq
                
                # Orphan 'curr' and link 'prev' to 'target'
                target.prev_ = prev_node
                prev_node.next_ = target
                
                # C. Add new born pairs to counter
                # New pair to the left: (Prev, NewSymbol)
                if prev_node and prev_node is not self.nodes:
                    self.persistent_counts[(prev_node.val, target.val)] += 1
                    new_f = self.persistent_counts[(prev_node.val, target.val)]
                    heapq.heappush(self.heapq_, (-new_f, (prev_node.val, target.val)))
                
                # New pair to the right: (NewSymbol, Next)
                if next_node:
                    self.persistent_counts[(target.val, next_node.val)] += 1
                    new_f = self.persistent_counts[(target.val, next_node.val)]
                    heapq.heappush(self.heapq_, (-new_f, (target.val, next_node.val)))
                
                # D. The pair we just merged (top_pair) is now gone
                self.persistent_counts[top_pair] -= 1
                
                self.num_nodes -= 1
                curr = target # Move to the new node to continue scanning
            else:
                curr = curr.next_

    def find_best(self):
        while self.heapq_:
            neg_freq, pair = heapq.heappop(self.heapq_)
            actual_freq = self.persistent_counts.get(pair, 0)
            
            # Check if this heap entry matches our Source of Truth
            if actual_freq == -neg_freq and actual_freq > 0:
                return pair, actual_freq
        return None, 0

    def GetFinalVocab(self):
        final_vocab = set()
        curr = self.nodes.next_ # Start after the head sentinel
        
        while curr.next_ is not None: # Stop before the tail sentinel
            # Converting the list seq to a tuple so it's hashable for the set
            final_vocab.add(tuple(curr.seq))
            curr = curr.next_
            
        return final_vocab

def print_vocab(vocab, initial_num_nodes, BMO):
    # watch out for global - initial_num_nodes
    for v in vocab:
        try:
            # Convert tuple of ints -> bytes -> string
            readable_str = bytes(v).decode('utf-8')
            # Use repr to see hidden characters like \n or \r
            print(f"{v} -> {repr(readable_str)}") 
        except UnicodeDecodeError:
            # This happens if a token is a partial byte sequence 
            # (common in mid-training BPE)
            print(f"{v} -> [Partial Multi-byte Sequence]")

    print(f"init num nodes: {initial_num_nodes}\nfinal num nodes:{BMO.num_nodes}")
    print(f"compression ratio: {BMO.num_nodes/ initial_num_nodes}")

def save_vocab_to_json(vocab, step, initial_nodes, current_nodes):
    output_filename = f"vocab_step_{step}.json"
    json_output = []
    
    for v in vocab:
        try:
            # decode('utf-8') handles the Cyrillic and English
            readable_str = bytes(v).decode('utf-8')
        except UnicodeDecodeError:
            readable_str = "[Partial Multi-byte Sequence]"
            
        json_output.append({
            "sequence": list(v), # JSON needs lists, not tuples
            "text": readable_str
        })

    # Wrap the data and stats into one object
    data_to_save = {
        "metadata": {
            "step": step,
            "initial_nodes": initial_nodes,
            "current_nodes": current_nodes,
            "compression_ratio": current_nodes / initial_nodes
        },
        "vocabulary": json_output
    }

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    
    print(f"--- Saved {output_filename} | Ratio: {current_nodes / initial_nodes:.4f} ---")

if  __name__ == '__main__':
    fname = "bookdat/Crime_and_Punishment_.txt"
    
    start_time = time.perf_counter()

    with open(fname, 'rb') as f:
        text_bytes  = f.read()        
    text_ints = list(map(int, text_bytes)) 
    print('init text lenght ', len(text_ints) )
    
    text_int_set = set(text_ints)
    print('initial token set ', text_int_set)
    print('len initial set ', len(text_int_set))
    
    # ok now the next step is to push elements to this chain class
    BMO = chain(text_ints)
    
    mid_time = time.perf_counter()

    initial_num_nodes = BMO.num_nodes
    steps = 512
    for i in range(steps):
        BMO.Update()

    vocab = BMO.GetFinalVocab()
    end_time = time.perf_counter()    
    
    print("\n--- FINAL VOCABULARY ---")
    print_vocab(vocab, initial_num_nodes, BMO)
    save_vocab_to_json(vocab, i, initial_num_nodes, BMO.num_nodes)

    # Calculations
    total_time = end_time - start_time
    setup_time = mid_time - start_time
    update_time = end_time - mid_time

    print("\n" + "="*30)
    print("       TIMING SUMMARY")
    print("="*30)
    print(f"Setup (Read + Build): {setup_time:10.4f}s")
    print(f"BPE Updates ({steps} steps): {update_time:10.4f}s")
    print(f"Total Execution:      {total_time:10.4f}s")
    print("-" * 30)
    print(f"Avg per Update Step:  {update_time/steps:10.6f}s")
    print(f"Compression Ratio:    {initial_num_nodes / BMO.num_nodes:10.2f}x")
    print("="*30)
