#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <set>
#include <chrono>
#include <queue>
using namespace std;

class Reader{
    public:
        long long char_count = 0; // doesnt need to be long long, but hey man large files . 
        string content;
        Reader(string filename_){
            ifstream inputFile(filename_);
            if (inputFile.is_open()){
                stringstream buffer;
                buffer<<inputFile.rdbuf();
                content = buffer.str();                
                // remember the line might be empty.... so cast it to prevent wrap around error
                // basically we manually itterate through every character in this thing and add it to a vector
                char_count = content.length();
                inputFile.close(); // close the file

                // cout<<content; // print the file 
        } else{
            cerr << "unable to open the file";
            }
        }
};

struct PairHasher {
    size_t operator()(const pair<int, int>& p) const {
        // Use a standard hash combiner for ints
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};

struct Node{
    // structs are public by default
    Node* prev_ = nullptr;
    Node* next_ = nullptr;
    int val;
    string seq;
    void printNode(){cout<<"val: "<<val<<"\nseq: "<<seq;}
};

class Chain{
    public:
        Node head;
        int num_nodes = 0;
        int init_num_nodes = 0;
        int symbols = 256;
        pair<int,int>bestPair = {-1,-1};
        unordered_map<pair<int, int>, int32_t, PairHasher> persistent_counts;

        // Stores (frequency, pair) for O(1) access to the max
        priority_queue<pair<int, pair<int, int>>> pq;

        Chain(string text){
            if (text.empty()) return;
            // construct the doubly linked list and count things
            
            // start the chain
            head.val = (uint8_t)text[0];
            head.seq = string(1, text[0]);

            Node* current = &head; // pointer to track the current node. right now, address of head
            num_nodes++;
            
            for (size_t i = 0; i < text.length() - 1; i++) {
                int v1 = (uint8_t)text[i];
                int v2 = (uint8_t)text[i+1];
                persistent_counts[{v1, v2}]++;
                pq.push({persistent_counts[{v1, v2}], {v1, v2}}); // make sure to push to priority queue
                
                // make a new node -- persist on heap with new
                Node* newNode = new Node(); // make a new node and point to it in 1 line
                newNode->val = v2;
                newNode->seq = string(1, text[i+1]);

                // link up 
                newNode->prev_ = current; // current is a pointer so this works
                current->next_ = newNode; // newNode is a pointer

                //update the current pointer to track the new node
                current = newNode;
                num_nodes++;
            }
            findBest();
            init_num_nodes+=num_nodes;
        }
        // deconstructor to free all the nodes 
        ~Chain() {
            Node* current = head.next_; // Start after the stack-allocated head
            while (current != nullptr) {
                Node* next = current->next_;
                delete current;
                current = next;
                }
            }  
        // helper function to find the best pair
        void findBest() {
            while (!pq.empty()) {
                auto top = pq.top();
                int freq = top.first;
                pair<int, int> p = top.second;

                // Lazy Deletion: Check if this heap entry is stale
                if (persistent_counts.count(p) && persistent_counts[p] == freq) {
                    bestPair = p;
                    return;
                }
                pq.pop(); // Remove stale or zero-count entries
            }
            bestPair = {-1, -1}; // No pairs left
        }
        void Update() {
            if (bestPair.first == -1) return;
            symbols++;

            Node* current = &head;
            while (current != nullptr && current->next_ != nullptr) {
                if (current->val == bestPair.first && current->next_->val == bestPair.second) {
                    Node* leftNode = current;
                    Node* rightNode = current->next_;

                    // --- 1. REMOVE OLD PAIRS (Surgical) ---
                    // Left-side connection
                    if (leftNode->prev_) {
                        pair<int, int> p = {leftNode->prev_->val, leftNode->val};
                        persistent_counts[p]--;
                        if (persistent_counts[p] == 0) persistent_counts.erase(p);
                    }
                    // Right-side connection
                    if (rightNode->next_) {
                        pair<int, int> p = {rightNode->val, rightNode->next_->val};
                        persistent_counts[p]--;
                        if (persistent_counts[p] == 0) persistent_counts.erase(p);
                    }
                    // The pair itself
                    persistent_counts[bestPair]--;
                    if (persistent_counts[bestPair] == 0) persistent_counts.erase(bestPair);

                    // --- 2. PERFORM MERGE ---
                    if (leftNode == &head) {
                        head.val = symbols;
                        head.seq += rightNode->seq;
                        Node* thirdNode = rightNode->next_;
                        head.next_ = thirdNode;
                        if (thirdNode) thirdNode->prev_ = &head;
                        
                        delete rightNode;
                        current = &head; 
                    } else {
                        rightNode->val = symbols;
                        rightNode->seq = leftNode->seq + rightNode->seq;
                        Node* before = leftNode->prev_;
                        rightNode->prev_ = before;
                        if (before) before->next_ = rightNode;
                        
                        delete leftNode;
                        current = rightNode;
                    }

                    // --- 3. ADD NEW PAIRS (Surgical) ---
                    // New left pair: (prev, new_symbol)
                    if (current->prev_) {
                        pair<int, int> p = {current->prev_->val, current->val};
                        persistent_counts[p]++;
                        pq.push({persistent_counts[p], p});
                    }
                    // New right pair: (new_symbol, next)
                    if (current->next_) {
                        pair<int, int> p = {current->val, current->next_->val};
                        persistent_counts[p]++;
                        pq.push({persistent_counts[p], p});
                    }

                    num_nodes--;
                } else {
                    current = current->next_;
                }
            }

            // After one full pass of merging bestPair, find the next one
            findBest(); 
            // cout << "Remaining Nodes: " << num_nodes << " | Current Symbol ID: " << symbols << endl;
        }
        
        void printUniqueSequences() {
            //set automatically handles uniqueness and sorting
            set<string> uniqueSeqs;

            // Start at the head and walk the chain
            Node* current = &head;
            while (current != nullptr) {
                uniqueSeqs.insert(current->seq);
                current = current->next_;
            }

            // Print the final vocabulary
            cout << "\n--- Unique Sequences Found (" << uniqueSeqs.size() << ") ---" << endl;
            for (const string& s : uniqueSeqs) {
                cout << "[" << s << "]" << endl;
            }
        }
    };

int main(){
    auto start_build = chrono::high_resolution_clock::now();
    string filename_ = "bookdat/Crime_and_Punishment_.txt";
    
    Reader BookWorm = Reader(filename_); // lol its data type is reader
    Chain BMO(BookWorm.content);
    auto end_build = chrono::high_resolution_clock::now();


    int steps = 512;
    auto start_updates = chrono::high_resolution_clock::now();
    for (int i =0; i<steps; i++){
        BMO.Update();
    }
    auto end_updates = chrono::high_resolution_clock::now();

    // Calculate durations
    chrono::duration<double> build_diff = end_build - start_build;
    chrono::duration<double> update_diff = end_updates - start_updates;

    BMO.printUniqueSequences();

    cout<<"init nodes : "<<BMO.init_num_nodes<<"\nFinal node count : "<<BMO.num_nodes
    <<"\ncompression ratio : "<<(float)BMO.num_nodes / (float)BMO.init_num_nodes<<"\n";
    
    cout << fixed << "\n--- Performance Results ---" << endl;
    cout << "Initial Build: " << build_diff.count() << " seconds" << endl;
    cout << "512 Updates:    " << update_diff.count() << " seconds" << endl;
    cout << "Avg per Update: " << (update_diff.count() / steps) << " seconds" << endl;
    
    return 0;
//should print out the 
}