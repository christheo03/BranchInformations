#include <iostream>
#include <fstream>
#include <unordered_map>
#include "pin.H"

using std::cerr;
using std::cout;
using std::dec;
using std::endl;
using std::hex;
using std::ofstream;
using std::unordered_map;

struct br_behavor
{
    UINT32 times_executed = 0;
    UINT32 times_taken = 0;
};
//Map to store <PC,(times_seen,times_taken)>
unordered_map<ADDRINT, br_behavor> br_info;

//Increase the times taken
VOID TakenCounter(ADDRINT instr)
{
    br_info[instr].times_taken++;
}

//Increase the times executed
VOID BranchCounter(ADDRINT instr)
{
    br_info[instr].times_executed++;
} 
//Print PC address, times executed and the times taken of the branches
VOID Fini(INT32 code, VOID *v)
{
    cout<<"Conditional Branches"<<endl;
    for (const auto &[add, info] : br_info)
    {
        cout<<"0x"<<hex<<add<<dec<< " Times: " << info.times_executed << " Taken: " << info.times_taken << endl;
    }
}

//Iterate thru the sections, routines of the sections and Instructions of the routines. Pin functions for conditional branches 

VOID ImageLoad(IMG img, VOID *v)
{
    //Work only with the executable seen in the command line (-- executable.exe)
    if (!IMG_IsMainExecutable(img))
    {
        return;
    }
    
    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
    {
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn))
        {

            RTN_Open(rtn);
            for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins))
            {
                //If the instructions is a conditional branch
                if (INS_Category(ins) == XED_CATEGORY_COND_BR)
                {
                    //Insert the BranchCount function before the instruction
                    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BranchCounter, IARG_INST_PTR, IARG_END);
                    //Insesrt this fucntion TakenCounter to the taken flow of the instuctions
                    INS_InsertCall(ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)TakenCounter, IARG_INST_PTR, IARG_END);
                }
            }
            RTN_Close(rtn);
        }
    }
}
// VOID Instruction(INS instr, VOID *v)
// {
//     if (IMG_IsMainExecutable(IMG_FindByAddress(INS_Address(instr))))
//     {
//         if (INS_Category(instr) == XED_CATEGORY_COND_BR)
//         {
//             INS_InsertCall(instr, IPOINT_BEFORE, (AFUNPTR)BranchCounter, IARG_INST_PTR, IARG_END);
//             INS_InsertCall(instr, IPOINT_TAKEN_BRANCH, (AFUNPTR)TakenCounter, IARG_INST_PTR, IARG_END);
//         }
//     }
// }

int main(int argc, char *argv[])
{
    PIN_Init(argc, argv);
    PIN_AddFiniFunction(Fini, 0);
    IMG_AddInstrumentFunction(ImageLoad, 0);
    // INS_AddInstrumentFunction(Instruction, 0);
    PIN_StartProgram(); 
}