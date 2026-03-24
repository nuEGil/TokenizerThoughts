#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <queue>
using namespace std;

/*
What we want here is what? 
learn the token set
write to file 
load a token set 
class for encoding and decoding

Maybe the main function has flags to turn this 
into a command line tool .

*/

class Reader {
    public:
        long long char_count = 0;
        string content;

    Reader(string filename_) {
        ifstream inputFile(filename_);
        if (inputFile.is_open()) {
            stringstream buffer;
            buffer << inputFile.rdbuf();
            content = buffer.str();
            char_count = content.length();
            inputFile.close();
        } else {
            cerr << "unable to open the file";
        }
    }
};

struct PairHasher {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};

// token set learning

struct dataArr {
    int32_t text_len;
    vector<int32_t> original_tok;
    vector<int32_t> tokens;
    vector<int32_t> prev;
    vector<int32_t> next;
    vector<bool>    alive;

    dataArr(size_t n)
        : original_tok(n), tokens(n), prev(n), next(n), alive(n),
          // FIX: cast to int32_t explicitly; note files >2GB will still
          // overflow - consider size_t if you need that range
          text_len(static_cast<int32_t>(n)) {}
};

class Chain {
    public:
        dataArr data;
        // FIX: start at 255 so the first merged token is assigned ID 256
        int symbols = 255;
        size_t text_len;

        pair<int,int> bestPair = {-1, -1};
        unordered_map<pair<int,int>, int32_t, PairHasher> persistent_counts;
        map<int, string> vocab_map;

        int32_t head_idx = 0;
        priority_queue<pair<int, pair<int,int>>> pq;

    Chain(string text) : data(text.length()) {
        if (text.empty()) return;
        // initializing the vocabulary
        for (int i = 0; i < 256; i++)
            vocab_map[i] = string(1, static_cast<unsigned char>(i));

        for (size_t i = 0; i < text.length(); i++) {
            data.original_tok[i] = static_cast<int32_t>(static_cast<unsigned char>(text[i]));
            data.tokens[i] = static_cast<int32_t>(static_cast<unsigned char>(text[i]));
            data.alive[i]  = true;
            data.prev[i]  = static_cast<int32_t>(i) - 1;
            data.next[i]  = (i == text.length() - 1) ? -1 : static_cast<int32_t>(i) + 1;

            if (i < text.length() - 1) {
                int v1 = static_cast<unsigned char>(text[i]);
                int v2 = static_cast<unsigned char>(text[i + 1]);
                persistent_counts[{v1, v2}]++;
            }
        }

        for (auto const& [pair, count] : persistent_counts)
            pq.push({count, pair});

        findBest();
    }

    void findBest() {
        while (!pq.empty()) {
            auto [freq, p] = pq.top();
            auto it = persistent_counts.find(p);
            if (it != persistent_counts.end() && it->second == freq) {
                if (freq <= 0) break;
                bestPair = p;
                return;
            }
            pq.pop();
        }
        bestPair = {-1, -1};
    }

    // FIX: helper to safely decrement a pair count and erase it when it
    // hits zero, preventing ghost counts and negative values
    void decrementPair(pair<int,int> key) {
        auto it = persistent_counts.find(key);
        if (it == persistent_counts.end()) return;
        it->second--;
        if (it->second <= 0)
            persistent_counts.erase(it);
    }

    void Update() {
        if (bestPair.first == -1) return;

        int32_t target_v1 = bestPair.first;
        int32_t target_v2 = bestPair.second;

        // FIX: pre-increment then assign, so ID 256 is the first merged token
        symbols++;
        vocab_map[symbols] = vocab_map[target_v1] + vocab_map[target_v2]; // this is where the vocabulary gets updated

        int32_t curr = head_idx;

        while (curr != -1) {
            int32_t L = curr;
            int32_t R = data.next[L];

            if (R != -1 && data.tokens[L] == target_v1 && data.tokens[R] == target_v2) {

                int32_t prev_idx = data.prev[L];
                int32_t next_idx = data.next[R];

                // FIX: use decrementPair() to avoid negative counts / ghost entries
                if (prev_idx != -1)
                    decrementPair({data.tokens[prev_idx], data.tokens[L]});
                if (next_idx != -1)
                    decrementPair({data.tokens[R], data.tokens[next_idx]});
                decrementPair({target_v1, target_v2});

                // Merge: R holds the new token, L is retired
                data.tokens[R] = symbols;
                data.alive[L]  = false;

                // Re-stitch linked list
                data.prev[R] = prev_idx;
                if (prev_idx != -1) {
                    data.next[prev_idx] = R;
                } else {
                    head_idx = R;
                }

                // Increment new neighbor pairs
                if (prev_idx != -1) {
                    pair<int,int> new_left = {data.tokens[prev_idx], symbols};
                    persistent_counts[new_left]++;
                    pq.push({persistent_counts[new_left], new_left});
                }
                if (next_idx != -1) {
                    pair<int,int> new_right = {symbols, data.tokens[next_idx]};
                    persistent_counts[new_right]++;
                    pq.push({persistent_counts[new_right], new_right});
                }

                curr = R;
            } else {
                curr = data.next[L];
            }
        }

        findBest();
    }

    string getSymbolString(int32_t tid) {
        auto it = vocab_map.find(tid);
        if (it != vocab_map.end()) return it->second;
        return "";
    }

    void printUniqueSequences() {
        set<string> uniqueSeqs;

        // FIX: start from head_idx, not 0; the old manual skip loop
        // was fragile when head_idx had advanced past index 0
        int32_t curr = head_idx;
        while (curr != -1) {
            uniqueSeqs.insert(getSymbolString(data.tokens[curr]));
            curr = data.next[curr];
        }

        cout << "\n--- Unique tokens in current chain ("
             << uniqueSeqs.size() << ") ---" << endl;
        for (const string& s : uniqueSeqs)
            cout << "[" << s << "]" << endl;
    }

    void printFullVocabulary() {
        cout << "\n--- Full learned vocabulary ("
             << vocab_map.size() << ") ---" << endl;
        for (auto const& [id, str] : vocab_map) {
            if (id >= 256)
                cout << "ID " << id << ": [" << str << "]" << endl;
        }
    }

    void getCompressionRatio() {
        int32_t alive_count = 0;
        for (int i = 0; i < data.text_len; i++) {
            if (data.alive[i]) alive_count++;
        }
        float ratio = static_cast<float>(alive_count)
                    / static_cast<float>(data.text_len);

        cout << "\n--- Compression stats ---" << endl;
        cout << "Original tokens:    " << data.text_len << endl;
        cout << "Compressed tokens:  " << alive_count << endl;
        cout << "Compression ratio:  " << ratio << endl;
    }
};

// Trie for encoding and decoding
struct TrieNode {
    // Array of pointers to each of 256 possible sub nodes.
    TrieNode* subs[256];
    int token_id;
    // isEndOfWord is true if the node represents the end of a word.
    bool isEndOfWord;

    // Constructor to initialize a new TrieNode.
    TrieNode() {
        // Initialize all subs pointers to nullptr.
        for (int i = 0; i < 256; i++) {
            subs[i] = nullptr;
        }
        // A new node is not the end of a word by default.
        isEndOfWord = false;
    }
};

// Trie class encapsulates the Trie data structure and its operations.
class Trie {
    private:
        TrieNode* root;

    public:
        // Constructor to initialize the Trie with an empty root node.
        // you're creating a new pointer on every method call... so like 1 byte per call. 
        // there's a way to do this with smart pointers to avaoid that, but later. 
        Trie() {
            root = new TrieNode();
        }

        // Inserts a word into the Trie.
        void insert(int id, string key) {
            TrieNode* current = root;
            for (char ch : key) {
                unsigned char index = (unsigned char)ch; // Safety for high ASCII
                if (current->subs[index] == nullptr) {
                    current->subs[index] = new TrieNode();
                }
                current = current->subs[index];
            }
            // Only set these at the very end of the word
            current->isEndOfWord = true;
            current->token_id = id;
        }

        vector<int> tokenize(const string& text){
            vector<int> output;
            int Pos = 0;

            while (Pos<text.length()){ // walk the text
                TrieNode* current = root;
                int bestTokenId = -1; 
                int bestLength = 0;

                for (int i = Pos; i<text.length(); i++){
                    unsigned char ch = (unsigned char) text[i];
                    if (current->subs[ch] == nullptr) break; // stop searching at a dead end

                    current = current ->subs[ch];
                    if (current->isEndOfWord){
                        bestTokenId = current->token_id;
                        bestLength = (i-Pos) + 1;
                        }
                    }

                if (bestLength > 0){
                    output.push_back(bestTokenId);
                    Pos += bestLength; // move the full length of the sequence forward
                } else{
                    Pos++; // move 1 forward
                }
            }
            return output;
        }
   
        // Recursive funciton to delete the Trie and free memory.
        void freeTrie(TrieNode* node) {
            if (!node) {
                return;
            }
            for (int i = 0; i < 256; i++) {
                freeTrie(node->subs[i]);
            }
            delete node;
        }
        
        // Destructor to clean up the allocated memory.
        ~Trie() {
            freeTrie(root);
        }
    };

int main() {
    auto start_build = chrono::high_resolution_clock::now();

    string filename_ = "bookdat/Crime_and_Punishment_.txt";
    Reader BookWorm(filename_);
    Chain  BMO(BookWorm.content);

    auto end_build = chrono::high_resolution_clock::now();

    int steps = 512;
    auto start_updates = chrono::high_resolution_clock::now();
    for (int i = 0; i < steps; i++)
        BMO.Update();
    auto end_updates = chrono::high_resolution_clock::now();

    chrono::duration<double> build_diff  = end_build  - start_build;
    chrono::duration<double> update_diff = end_updates - start_updates;

    BMO.printFullVocabulary();
    cout << "Vocab map size: " << BMO.vocab_map.size() << endl;

    cout << fixed << "\n--- Performance results ---" << endl;
    cout << "Initial build:  " << build_diff.count()  << " s" << endl;
    cout << steps << " updates:  "
         << update_diff.count() << " s" << endl;
    cout << "Avg per update: "
         << (update_diff.count() / steps) << " s" << endl;

    BMO.getCompressionRatio();
    cout << "Total symbols: " << BMO.symbols << endl;

    // so now for the next part we need to go through BMO's vocabulary and turn that into a trie
    Trie VTrie = Trie();

    // prefered way to iterate through maps like this. 
    for (const auto& [id, word] : BMO.vocab_map){
        VTrie.insert(id, word);
        // cout<<"id :"<<id<<" word: "<<word<<"\n"; // printing out the full vocab cause why not. 
    }
    // ok now once we've inserted all the words.... you can write a method to encode
    vector<int> tokens = VTrie.tokenize(BookWorm.content);

    // Verification loop
    string decoded = "";
    for (int id : tokens) {
        decoded += BMO.getSymbolString(id);
    }

    if (decoded == BookWorm.content) {
        cout << "Success! Tokenization is lossless." << endl;
    }

    return 0;
}