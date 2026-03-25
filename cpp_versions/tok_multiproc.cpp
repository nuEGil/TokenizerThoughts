#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <queue>
#include <unistd.h>
#include <sys/wait.h>
using namespace std;

/*
What we want here is what? 
learn the token set
write to file 
load a token set 
class for encoding and decoding

Maybe the main function has flags to turn this 
into a command line tool .

then think about this thing 
string getSymbolString(int32_t tid) -- this should be independant of chain
if you want to just load up a vocabulary

so now we have to design some of that bit. 
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

// token set learning
struct PairHasher {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};

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


        // this should be something that is independent of the chain... because, what if i just load a vocabulary
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
// end token set learning

// Encoding and decoding
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

// end encoding and decoding

map<int, string> processBatch(const vector<string>& batch, int workerID) {
    map<int, string> output;
    unordered_set<string> seen; // fast look up
    int subGlobID = 0;

    for (const auto& path : batch) {
        Reader BookWorm(path);
        Chain BMO(BookWorm.content);

        for (int i = 0; i < 512; i++) BMO.Update();

        for (const auto& [localID, word] : BMO.vocab_map) {
            // .insert().second is true only if the word is NEW
            if (seen.insert(word).second) {
                output[subGlobID] = word;
                subGlobID++;
            }
        }
    }
    return output;
}
    
void sendMap(int pipe_fd, const map<int, string>& vocab) {
    size_t mapSize = vocab.size();
    if (write(pipe_fd, &mapSize, sizeof(mapSize)) == -1) {
        perror("write mapSize failed");
        return;
    }

    for (const auto& [id, word] : vocab) {
        if (write(pipe_fd, &id, sizeof(id)) == -1) break;
        
        size_t strLen = word.length();
        if (write(pipe_fd, &strLen, sizeof(strLen)) == -1) break;
        if (write(pipe_fd, word.data(), strLen) == -1) break;
    }
}

map<int, string> receiveMap(int pipe_fd) {
    map<int, string> vocab;
    size_t mapSize = 0;

    // 1. Read the map size. Check if the pipe actually gave us data.
    if (read(pipe_fd, &mapSize, sizeof(mapSize)) <= 0) return vocab;

    for (size_t i = 0; i < mapSize; ++i) {
        int id;
        size_t strLen;

        // 2. Read metadata (ID and Length)
        if (read(pipe_fd, &id, sizeof(id)) <= 0) break;
        if (read(pipe_fd, &strLen, sizeof(strLen)) <= 0) break;

        // 3. Prepare the string buffer
        string word(strLen, '\0');
        
        // 4. Handle the string data
        size_t totalRead = 0;
        while (totalRead < strLen) {
            ssize_t result = read(pipe_fd, &word[totalRead], strLen - totalRead);
            if (result <= 0) break; // Pipe closed or error
            totalRead += result;
        }

        vocab[id] = word;
    }
    return vocab;
}

//check this out later.... interesting. 
void saveVocabBinary(const string& filename, const map<string, int>& vocab) {
    ofstream outFile(filename, ios::binary);
    if (!outFile) return;

    size_t vocabSize = vocab.size();
    outFile.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));

    for (const auto& [word, id] : vocab) {
        // Write ID
        outFile.write(reinterpret_cast<const char*>(&id), sizeof(id));
        // Write String length and then data
        size_t len = word.length();
        outFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
        outFile.write(word.data(), len);
    }
    outFile.close();
}


int main() {
    // byte pair encoding can handle different languages! lol. 
    vector<string> paths ={
        "bookdat/Bhagavad-Gita_(Besant_4th).txt",
        "bookdat/Crime_and_Punishment_.txt",
        "bookdat/The_Bhagavad_Gita_(Arnold_translation).txt",
        "bookdat/The_Brothers_Karamazov.txt",
        "bookdat/The_Gambler_(1867).txt",
        "bookdat/The_Jungle.txt",
        "bookdat/Unready_to_Wear.txt",
        "bookdat/War_and_Peace_(Tolstoy).txt",
        "bookdat/Pride_and_Prejudice_(1813).txt",
        };

    int  numWorkers = 4;
    int batchSize = paths.size() / numWorkers;
    
    // creating pipes for workers: pipefds[workerid][0] is read, [1] is write
    int pipefds[4][2];

    // pipe initialization 
    for (int i = 0; i < numWorkers; i++) {
        if (pipe(pipefds[i]) == -1) {
            perror("Pipe initialization failed");
            return 1;
        }
    }

    for (int i = 0; i <numWorkers; i++){
        auto start = paths.begin() + (i* batchSize);
        auto end = (i == numWorkers - 1) ? paths.end() : start + batchSize;
        vector<string> myBatch(start, end) ; //  vector definition by sub sampling another vector
    
        pid_t pid = fork();

        if (pid < 0){
            perror("Fork failed");
            return 1;
        }
        if (pid == 0) {
            // 1. Close the read end—the child only writes
            close(pipefds[i][0]); 

            // 2. Run the batch processor (now returning a single map)
            map<int, string> workerResult = processBatch(myBatch, i);

            // 3. Send the single merged map across the pipe
            sendMap(pipefds[i][1], workerResult);

            // 4. Cleanup and exit
            close(pipefds[i][1]);
            _exit(0);
        } else{
            close(pipefds[i][1]); // main process wont write to this pipe
        }
    }

    // collect
    map<string, int> globalVocab;
    int nextGlobalID = 0;

    for (int i = 0; i < numWorkers; i++) {
        // Receive the single merged map from this worker
        map<int, string> localMap = receiveMap(pipefds[i][0]);

        // Merge into the global vocabulary
        for (const auto& [localID, word] : localMap) {
            if (globalVocab.insert({word, nextGlobalID}).second) {
                nextGlobalID++;
            }
        }
        
        close(pipefds[i][0]);
        wait(nullptr); 
    }

    
    for (const auto&  [word, id] : globalVocab){
        cout<<"id: "<<id<<" word: "<<word<<"\n";
    }
    cout << "Successfully merged " << paths.size() << " books.\n";
    cout << "Unique Global Tokens: " << globalVocab.size() << endl;


    saveVocabBinary("bookdat/vocab.bin", globalVocab);

    return 0;
}