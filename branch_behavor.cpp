#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <iomanip>
#include "pin.H"

using std::cerr;
using std::cout;
using std::dec;
using std::endl;
using std::hex;
using std::left;
using std::map;
using std::ofstream;
using std::setw;
using std::string;
#define NUM_PREV 10
UINT64 total_executions = 0;
UINT64 total_taken = 0;

struct instruction
{
    UINT32 opcode = 0;
    USIZE size = 0;
};

struct flag_instruction
{
    OPCODE opcode = 0;
    USIZE size = 0;
    UINT RA_type;
    UINT RB_type;
};

struct branch
{
    UINT32 times_executed = 0;
    instruction prev_instr[NUM_PREV];
    instruction next_instr[NUM_PREV];
    string flag_wr_instr;
    UINT32 times_taken = 0;
    string opcode;
    INT32 offset = 0;
    USIZE size = 0;
};

// Map to store < PC, (times_seen, times_taken, opcode) >
map<ADDRINT, branch> br_info;

// Increase the times taken
VOID TakenCounter(ADDRINT instr)
{
    br_info[instr].times_taken++;
    total_taken++;
}

// Increase the times executed
VOID BranchCounter(ADDRINT instr)
{
    total_executions = total_executions + 1;
    br_info[instr].times_executed++;
}

// Print PC address, times executed and the times taken of the branches

VOID Fini(INT32 code, VOID *v)
{
    ofstream output("branches.out");

    output << "Conditional Executions: " << total_executions << endl;
    output << "Total Taken: " << total_taken << endl;

    // Remove non-executed branches
    for (auto it = br_info.begin(); it != br_info.end();)
    {
        if (it->second.times_executed == 0)
            it = br_info.erase(it);
        else
            ++it;
    }

    // ---- HEADER ----
    output << left
           << setw(20) << "Address"
           << setw(10) << "Opcode"
           << setw(10) << "Executed"
           << setw(10) << "Taken"
           << setw(20) << "Offset"
           << setw(7) << "Size"
           << setw(39) << "Flag_Write_Instruction";

    // Generate "Prev1 Prev2 ... PrevN"
    for (int i = 0; i < NUM_PREV; i++)
    {
        output << setw(10) << ("Prev Op(" + std::to_string(i + 1) + ")");
        output << setw(10) << ("Size");
    }
    for (int i = 0; i < NUM_PREV; i++)
    {
        output << setw(10) << ("Next Op(" + std::to_string(i + 1) + ")");
        output << setw(10) << ("Size");
    }

    output << endl;

    // ---- ROWS ----
    for (const auto &[addr, info] : br_info)
    {
        output << left
               << "0x" << setw(20) << std::hex << addr << std::dec
               << setw(10) << info.opcode
               << setw(10) << info.times_executed
               << setw(10) << info.times_taken
               << setw(20) << info.offset
               << setw(7) << info.size
               << setw(39) << info.flag_wr_instr;

        // Print the prev instructions using a loop
        for (int i = 0; i < NUM_PREV; i++)
        {
            output << setw(10) << info.prev_instr[i].opcode << setw(10) << info.prev_instr[i].size;
        }

        for (int i = 0; i < NUM_PREV; i++)
        {
            output << setw(10) << info.next_instr[i].opcode << setw(10) << info.next_instr[i].size;
        }

        output << endl;
    }

    output.close();
}

// Iterate thru the sections, routines of the sections and Instructions of the routines. Pin functions for conditional branches
VOID ImageLoad(IMG img, VOID *v)
{

    // Work only with the executable seen in the command line (-- executable.exe)

    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
    {
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn))
        {
            INS flag_write = INS_Invalid();
            RTN_Open(rtn);
            for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins))
            {

                if (INS_RegWContain(ins, REG_GFLAGS))
                {
                    flag_write = ins;
                }
                // If the instructions is a conditional branch
                if (INS_Category(ins) == XED_CATEGORY_COND_BR)
                {
                    ADDRINT addr = INS_Address(ins);
                    br_info[addr].opcode = OPCODE_StringShort(INS_Opcode(ins));
                    if (INS_IsDirectControlFlow(ins))
                    {
                        br_info[addr].offset = INS_DirectControlFlowTargetAddress(ins) - addr;
                    }
                    br_info[addr].size = INS_Size(ins);
                    int i = 0;

                    for (INS prev = INS_Prev(ins); INS_Valid(prev) && i < NUM_PREV; prev = INS_Prev(prev))
                    {
                        br_info[addr].prev_instr[i].opcode = INS_Opcode(prev);
                        br_info[addr].prev_instr[i].size = INS_Size(prev);
                        i = i + 1;
                    }
                    i = 0;
                    for (INS next = INS_Next(ins); INS_Valid(next) && i < NUM_PREV; next = INS_Next(next))
                    {
                        br_info[addr].next_instr[i].opcode = INS_Opcode(next);
                        br_info[addr].next_instr[i].size = INS_Size(next);
                        i = i + 1;
                    }

                    if (INS_Valid(flag_write))
                    {
                        br_info[addr].flag_wr_instr = INS_Disassemble(flag_write);
                    }
                    else
                    {
                        br_info[addr].flag_wr_instr = "--";
                    }
                    // Insert the BranchCount function before the instruction
                    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BranchCounter, IARG_INST_PTR, IARG_END);
                    // Insesrt this fucntion TakenCounter to the taken flow of the instuctions
                    INS_InsertCall(ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)TakenCounter, IARG_INST_PTR, IARG_END);
                }
            }
            RTN_Close(rtn);
        }
    }
}

int main(int argc, char *argv[])
{
    PIN_Init(argc, argv);
    PIN_AddFiniFunction(Fini, 0);
    IMG_AddInstrumentFunction(ImageLoad, 0);
    PIN_StartProgram();
}