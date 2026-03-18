# make file for the token set learner
CXX = g++
CXXFLAGS = -std=c++20 
CXXFLAGS_fast = -std=c++20 -O3
SRC_DIR = .
build-all: tok_chain tok_chain_fast tok_array_fast 

tok_chain:
	@echo "Compling tok chain"
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/tok_chain2.cpp -o $(SRC_DIR)/tok_chain.exe 
	@echo "base verion built"

tok_chain_fast:
	@echo "Compling tok chain fast"
	$(CXX) $(CXXFLAGS_fast) $(SRC_DIR)/tok_chain2.cpp -o $(SRC_DIR)/tok_chain_fast.exe 
	@echo "fast verion built"

tok_array_fast:
	@echo "Compling tok array fast"
	$(CXX) $(CXXFLAGS_fast) $(SRC_DIR)/tok_arr2.cpp -o $(SRC_DIR)/tok_arr_fast.exe 
	@echo "fast array verion built"
