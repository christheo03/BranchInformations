#include "pin.H"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <algorithm>

using std::cerr;
using std::cout;
using std::endl;
using std::hex;
using std::dec;
using std::getline;
using std::ifstream;
using std::ostringstream;
using std::string;
using std::vector;
using std::stringstream;
using std::map;

enum SuccKind{
    SUCC_TAKEN,
    SUCC_FALL
};

struct Succ{
    ADDRINT branch_addr;
    SuccKind kind;
    vector <string> tracked_regs;
};

map<ADDRINT,vector<Succ>> succ_map;

string Trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos)
        return "";

    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

vector<string> SplitCsvLine(const string& line){

    vector<string> cols;
    stringstream ss(line);
    string item;
    while(getline(ss,item,',')){
        cols.push_back(Trim(item));
    }
    return cols;
}

vector<string> ParseRegsRead(const string& regsField) {
    vector<string> regs;
    stringstream ss(regsField);
    string token;

    while (ss >> token) {
        size_t pos = token.find('(');
        string reg = token.substr(0, pos);

        bool exists = false;
        for (const auto& r : regs) {
            if (r == reg) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            regs.push_back(reg);
        }
    }

    return regs;
}

bool Load_CSV(const string& path){
    ifstream file(path);
    if(!file.is_open()){
        cerr<<"ERROR: Could not open CSV file: "<< path <<"\n";
        return false;
    }

    string line;

    if(!std::getline(file,line)){
        cerr<<"ERROR: CSV is empty\n";
        return false;
    }

    vector<string> header = SplitCsvLine(line);

    int idxAddress = -1;
    int idxRegsRead = -1;
    int idxTaken = -1;
    int idxFall = -1;

    for (int i = 0; i < (int)header.size(); i++) {
        if (header[i] == "Address") idxAddress = i;
        else if (header[i] == "Regs_Read") idxRegsRead = i;
        else if (header[i] == "taken_bb_addr") idxTaken = i;
        else if (header[i] == "fall_bb_addr") idxFall = i;
    }

    if (idxAddress == -1 || idxRegsRead == -1 || idxTaken == -1 || idxFall == -1) {
        cerr << "ERROR: Missing required columns in CSV\n";
        return false;
    }



    while (getline(file, line)) {
        if (Trim(line).empty()) continue;
        vector<string> cols = SplitCsvLine(line);

        ADDRINT branch_addr = std::stoull(cols[idxAddress], nullptr, 0);
        ADDRINT taken_addr = 0;
        ADDRINT fall_addr = 0;

        if (cols[idxTaken] != "-1")
            taken_addr = std::stoull(cols[idxTaken], nullptr, 0);

        if (cols[idxFall] != "-1")
            fall_addr = std::stoull(cols[idxFall], nullptr, 0);

        vector<string> regs = ParseRegsRead(cols[idxRegsRead]);
        if (taken_addr != 0) {
            Succ s;
            s.branch_addr = branch_addr;
            s.kind = SUCC_TAKEN;
            s.tracked_regs = regs;
            succ_map[taken_addr].push_back(s);
        }

        if (fall_addr != 0) {
            Succ s;
            s.branch_addr = branch_addr;
            s.kind = SUCC_FALL;
            s.tracked_regs = regs;
            succ_map[fall_addr].push_back(s);
        }


    }
    return true;
}

VOID Trace(TRACE trace, VOID *v) {
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {
        ADDRINT bbl_addr = BBL_Address(bbl);

        auto it = succ_map.find(bbl_addr);
        if (it != succ_map.end()) {
            cout << "Matched BBL: 0x" << std::hex << bbl_addr << std::dec << "\n";

            for (const auto& s : it->second) {
                cout << "  branch_addr: 0x" << std::hex << s.branch_addr << std::dec << "\n";
                cout << "  kind: " << (s.kind == SUCC_TAKEN ? "TAKEN" : "FALL") << "\n";
                cout << "  tracked_regs: ";

                for (const auto& reg : s.tracked_regs) {
                    cout << reg << " ";
                }
                cout << "\n";
            }

            cout << "\n";
        }
    }
}


int main(int argc, char* argv[]) {
    if (PIN_Init(argc, argv)) {
        cerr << "PIN_Init failed\n";
        return 1;
    }
    string path= "branches.csv";

    if(!Load_CSV(path)){
        return 1;
    }
    cout<<"Loaded CSV Succesfull\n";

    TRACE_AddInstrumentFunction(Trace, 0);
    PIN_StartProgram();
    return 0;

}