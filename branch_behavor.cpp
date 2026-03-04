#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <iomanip>
#include <vector>
#include "pin.H"

using std::cerr;
using std::cout;
using std::endl;
using std::map;
using std::ofstream;
using std::setw;
using std::string;
using std::vector;

#define NUM_PREV 5

UINT64 total_executions = 0;
UINT64 total_taken = 0;
UINT64 bbl_id = 0;

struct regs
{
    REG reg;
    ADDRINT define_instr_pc=0;
    OPCODE define_instr_op=XED_ICLASS_INVALID;
};

//Simple Instruction Struct for Prevs and Next instructions
struct instruction
{
    UINT32 opcode;
    USIZE size;
};

//Struct for flag_write instructions
struct flag_instruction
{
    ADDRINT pc;
    OPCODE opcode = 0;
    BOOL same_bbl = false;
    USIZE size = 0;
    vector<regs> reg_read;
    vector<regs> reg_write;
};

// Struct for branch Instructions
struct branch
{
    UINT32 times_executed = 0;
    instruction prev_instr[NUM_PREV];
    instruction next_instr[NUM_PREV];
    flag_instruction flag_wr_instr;
    UINT32 times_taken = 0;
    string opcode;
    INT32 offset = 0;
    USIZE size = 0;
    string routine_type;
};


map<ADDRINT, branch> br_info; // Map for branch's heuristics
map<ADDRINT, UINT32> bbls; // Map that shows in what basic block an instruction is located
// Increase the times taken
VOID TakenCounter(ADDRINT instr)
{
    br_info[instr].times_taken++;
    total_taken++;
}

// Increase the times executed
VOID BranchCounter(ADDRINT instr)
{
    total_executions++;
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

    output << std::left
           << setw(20) << "Address"
           << setw(10) << "Opcode"
           << setw(10) << "Executed"
           << setw(10) << "Taken"
           << setw(20) << "Offset"
           << setw(7) << "Size"
           << setw(18) << "Flag_Write_PC"
           << setw(18) << "Flag_Instr_Opcode"
           << setw(18) << "same_BBL"
           << setw(15) << "Routine_Type"
           << setw(40) << "Regs_read"
           << setw(15) << "Regs_write";
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

    for (auto &[addr, info] : br_info)
    {
        info.flag_wr_instr.same_bbl = (bbls[addr] == bbls[info.flag_wr_instr.pc]);

        output << std::left
               << "0x" << setw(20) << std::hex << addr << std::dec
               << setw(10) << info.opcode
               << setw(10) << info.times_executed
               << setw(10) << info.times_taken
               << setw(20) << info.offset
               << setw(7) << info.size
               << "0x" << setw(18) << std::hex << info.flag_wr_instr.pc << std::dec
               << setw(18) << info.flag_wr_instr.opcode
               << setw(18) << info.flag_wr_instr.same_bbl
               << setw(15) << info.routine_type;
        std::ostringstream rr, rw;
        for (auto r : info.flag_wr_instr.reg_read){
            rr << REG_StringShort(r.reg) << "(";
            if(bbls[r.define_instr_pc]!=bbls[addr]){
                rr << "-1";
            }
            else{
                rr << OPCODE_StringShort(r.define_instr_op);
            }
            rr << ") ";
        }

        for (auto r : info.flag_wr_instr.reg_write)
            rw << REG_StringShort(r.reg) << " ";

        output << setw(40) << rr.str()
               << setw(15) << rw.str();
        
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

    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
    {
        for (RTN rtn = SEC_RtnHead(sec); RTN_Valid(rtn); rtn = RTN_Next(rtn))
        {
            INS flag_write = INS_Invalid();
            RTN_Open(rtn);
            ADDRINT rtn_entry = RTN_Address(rtn);

            bool hasCall=false;
            bool isRecursive=false;
            vector<ADDRINT> br_in_rtn; // Keep track of branches in THIS routine
            for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins))
            {
                if (INS_IsCall(ins)) {
                    hasCall = true;
                    if (INS_IsDirectControlFlow(ins) && (INS_DirectControlFlowTargetAddress(ins) == rtn_entry)) {
                        isRecursive = true;
                    }
                }
                

                if (INS_RegWContain(ins, REG_GFLAGS)) // Check is the Instruction writes to flag registers
                {
                    flag_write = ins;
                }
                // If the instructions is a conditional branch
                if (INS_Category(ins) == XED_CATEGORY_COND_BR)
                {
                    ADDRINT addr = INS_Address(ins);
                    br_in_rtn.push_back(addr); // Record this branch to label later

                    br_info[addr].opcode = OPCODE_StringShort(INS_Opcode(ins)); // Opcode of branch
                    if (INS_IsDirectControlFlow(ins))
                    {
                        br_info[addr].offset = INS_DirectControlFlowTargetAddress(ins) - addr; // Branch's Offset 
                    }
                    br_info[addr].size = INS_Size(ins); // Branch's instruction size 
                    int i = 0;

                    // Previous N Instructions from branch
                    for (INS prev = INS_Prev(ins); INS_Valid(prev) && i < NUM_PREV; prev = INS_Prev(prev))
                    {
                        br_info[addr].prev_instr[i].opcode = INS_Opcode(prev); 
                        br_info[addr].prev_instr[i].size = INS_Size(prev);
                        i = i + 1;
                    }
                    i = 0;
                    // Next N Instructions from branch
                    for (INS next = INS_Next(ins); INS_Valid(next) && i < NUM_PREV; next = INS_Next(next))
                    {
                        br_info[addr].next_instr[i].opcode = INS_Opcode(next);
                        br_info[addr].next_instr[i].size = INS_Size(next);
                        i = i + 1;
                    }


                    if (INS_Valid(flag_write))
                    {
                        br_info[addr].flag_wr_instr.opcode = INS_Opcode(flag_write); // Opcode of the Flag_Write Instr
                        br_info[addr].flag_wr_instr.pc = INS_Address(flag_write); // PC of Flag_Write Instr
                        br_info[addr].flag_wr_instr.size = INS_Size(flag_write); // Size of Flag_Write Instr

                        for (UINT32 i = 0; i < INS_MaxNumRRegs(flag_write); i++)
                        {
                            REG r = INS_RegR(flag_write, i);  // Register that the Flag_Written Instruction reads
                            regs new_reg;
                            new_reg.reg= REG_FullRegName(r);
                            INS prev= INS_Prev(flag_write);
                            while(INS_Valid(prev)){
                                for (UINT32 w=0; w<INS_MaxNumWRegs(prev);w++){
                                    REG wreg=INS_RegW(prev,w);
                                    if(wreg == REG_INVALID()) continue;

                                    if(REG_FullRegName(wreg) == new_reg.reg)
                                    {
                                        new_reg.define_instr_pc = INS_Address(prev);
                                        new_reg.define_instr_op=INS_Opcode(prev);
                                        break;
                                    }
                                }
                                if (new_reg.define_instr_pc != 0)
                                break;

                                prev=INS_Prev(prev);
                            }                       
                            br_info[addr].flag_wr_instr.reg_read.push_back(new_reg);
                        }

                        for (UINT32 i = 0; i < INS_MaxNumWRegs(flag_write); i++)
                        { 
                            REG r = INS_RegW(flag_write, i); // Register that Flag_write Instuction writes
                            regs new_reg;
                            new_reg.reg=r;
                            br_info[addr].flag_wr_instr.reg_write.push_back(new_reg);
                        }
                    }
                    else
                    {
                        br_info[addr].flag_wr_instr.pc = -1;
                    }
                    // Insert the BranchCount function before the instruction
                    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BranchCounter, IARG_INST_PTR, IARG_END);
                    // Insesrt this fucntion TakenCounter to the taken flow of the instuctions
                    INS_InsertCall(ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)TakenCounter, IARG_INST_PTR, IARG_END);
                }
            }
        
            string type = isRecursive ? "Recursive" : (hasCall ? "NonLeaf" : "Leaf");
            for (ADDRINT b_addr : br_in_rtn) {
                br_info[b_addr].routine_type = type;
            }
            RTN_Close(rtn);

        }
        
    }
}


// Function that maps the instruction with its basic block
// When does the Trace calls
void Trace(TRACE trace, VOID *v)
{

    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl))
    {

        for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins))
        {
            bbls[INS_Address(ins)] = bbl_id;
            
        } 
        bbl_id++;
    }
}

int main(int argc, char *argv[])
{
    PIN_InitSymbols();
    PIN_Init(argc, argv);
    PIN_AddFiniFunction(Fini, 0);
    TRACE_AddInstrumentFunction(Trace, 0);
    IMG_AddInstrumentFunction(ImageLoad, 0);
    PIN_StartProgram();
}