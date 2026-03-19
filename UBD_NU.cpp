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

struct BranchResult{
    int taken=-1;
    int fall=-1;
};

map<ADDRINT,vector<Succ>> succ_map;
map<ADDRINT,BranchResult> branch_results;

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
        if(it==succ_map.end())continue;

        for (const auto& s : it ->second){
            bool ubd=false;

            for (const auto& tracked_reg : s.tracked_regs){
                bool reads= false;
                bool writes=false;
                for(INS ins=BBL_InsHead(bbl); INS_Valid(ins); ins=INS_Next(ins)){

                    UINT32 rcount = INS_MaxNumRRegs(ins);
                    for(UINT32 i=0; i<rcount; i++){
                        REG r= INS_RegR(ins,i);
                        if(REG_StringShort(r)== tracked_reg){
                            if(writes){
                                break;
                            }
                            reads=true;
                            break;
                        }
                    }
                    UINT32 wcount = INS_MaxNumWRegs(ins);
                    for (UINT32 i = 0; i < wcount; i++) {
                        REG r = INS_RegW(ins, i);
                        if (REG_StringShort(r) == tracked_reg) {
                            writes=true;
                            if(reads) ubd=true;
                            }
                        }
                }
                if (ubd) break;
            }
            if (s.kind == SUCC_TAKEN)
                branch_results[s.branch_addr].taken = (ubd ? 1 : 0);
            else
                branch_results[s.branch_addr].fall = (ubd ? 1 : 0);
        }
    }
}

VOID Fini(INT32 code, VOID *v) {
    std::ofstream out("ubd_results.csv");

    out << "Address,taken_ubd,fall_ubd\n";

    for (const auto& entry : branch_results) {
        ADDRINT branch_addr = entry.first;
        const BranchResult& res = entry.second;

        out << "0x" << std::hex << branch_addr << std::dec
            << "," << res.taken
            << "," << res.fall
            << "\n";
    }

    out.close();
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
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;

}