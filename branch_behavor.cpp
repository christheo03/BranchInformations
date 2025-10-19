#include <iostream>
#include <fstream>
#include <unordered_map>
#include "pin.H"

using std::ofstream;
using std::endl;
using std::cerr;
using std::cout;
using std::unordered_map;

struct br_behavor{
    UINT32 times_seen=0;
    UINT32 times_taken=0;
};
// static UINT64 in_count=0;
unordered_map<ADDRINT,br_behavor> br_info;

VOID Fini(INT32 code, VOID *v){
for (const auto &[addr, info] : br_info) {
    cout << "0x" << std::hex << addr << std::dec
         << " | times_seen: " << info.times_seen
         << " | times_taken: " << info.times_taken << endl;
}
}
VOID Instruction(INS instr, VOID *v){
    if(INS_Category(instr)==XED_CATEGORY_COND_BR){
        ADDRINT address=INS_Address(instr);
        br_behavor &br=br_info[address];
        br.times_seen++;
        
    }

}

int main(int argc,char *argv[]){
    PIN_Init(argc,argv);
    PIN_AddFiniFunction(Fini,0);
    INS_AddInstrumentFunction(Instruction,0);
    PIN_StartProgram();
}