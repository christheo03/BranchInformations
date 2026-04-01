#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <iomanip>
#include <vector>
#include "pin.H"
#include <unordered_map>

using std::cerr;
using std::cout;
using std::endl;
using std::unordered_map;
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

unordered_map<ADDRINT, branch> br_info; // Map for branch's heuristics
unordered_map<ADDRINT, UINT32> bbls; // Map that shows in what basic block an instruction is located
unordered_map<REG, ADDRINT> last_writer_pc; // key register value address of instruction
unordered_map<REG, OPCODE> last_writer_op; // key register value opcode of instruction

VOID PIN_FAST_ANALYSIS_CALL TakenCounter(branch *b){
    b->times_taken++;
    total_taken++;
}

VOID PIN_FAST_ANALYSIS_CALL BranchCounter(branch *b){
    total_executions++;
    b->times_executed++;
}
// Increase the times taken
// VOID TakenCounter(ADDRINT instr)
// {
//     br_info[instr].times_taken++;
//     total_taken++;
// }

// Increase the times executed
// VOID BranchCounter(ADDRINT instr)
// {
//     total_executions++;
//     br_info[instr].times_executed++;
// }

// Print PC address, times executed and the times taken of the branches

VOID Fini(INT32 code, VOID *v)
{

    ofstream output("branches.csv");

    output << "#Conditional Executions: " << total_executions << "\n";
    output << "#Total Taken: " << total_taken << "\n";


    output << "Address,Opcode,Executed,Taken,Offset,Size,Flag_Write_PC,"
           << "Flag_Instr_Opcode,Same_BBL,Routine_Type,Regs_Read,Regs_Write";
    
    for (int i = 0; i < NUM_PREV; i++) {
        output << ",Prev_Op_" << (i + 1) << ",Prev_Size_" << (i + 1);
    }
    for (int i = 0; i < NUM_PREV; i++) {
        output << ",Next_Op_" << (i + 1) << ",Next_Size_" << (i + 1);
    }
    output << "\n";


    for (auto it = br_info.begin(); it != br_info.end(); ++it)
    {
        auto addr = it->first;
        auto &info = it->second;
        // Skip non-executed branches
        if (info.times_executed == 0) continue;

        // Calculate BBL logic
        info.flag_wr_instr.same_bbl = (bbls[addr] == bbls[info.flag_wr_instr.pc]);

        // Basic Info
        output << "0x" << std::hex << addr << std::dec << ","
               << info.opcode << ","
               << info.times_executed << ","
               << info.times_taken << ","
               << info.offset << ","
               << info.size << ","
               << "0x" << std::hex << info.flag_wr_instr.pc << std::dec << ","
               << info.flag_wr_instr.opcode << ","
               << info.flag_wr_instr.same_bbl << ","
               << info.routine_type << ",";

        // Regs Read (Joined by semicolons to keep them in one CSV cell)
        output << "\"";
        for (auto r : info.flag_wr_instr.reg_read) {
            output << REG_StringShort(r.reg) << "(";
            if (bbls[r.define_instr_pc] != bbls[addr]) {
                output << "-1";
            } else {
                output << OPCODE_StringShort(r.define_instr_op);
            }
            output << ") ";
        }
        output << "\",";

        // Regs Write
        output << "\"";
        for (auto r : info.flag_wr_instr.reg_write) {
            output << REG_StringShort(r.reg) << " ";
        }
        output << "\",";

        // Prev Instructions
        for (int i = 0; i < NUM_PREV; i++) {
            output << info.prev_instr[i].opcode << "," << info.prev_instr[i].size << ",";
        }

        // Next Instructions
        for (int i = 0; i < NUM_PREV; i++) {
            output << info.next_instr[i].opcode << "," << info.next_instr[i].size;
            if (i < NUM_PREV - 1) output << ","; // No comma after the very last element
        }

        output << "\n";
    }

    output.close();
}

// Iterate thru the sections, routines of the sections and Instructions of the routines. Pin functions for conditional branches
VOID ImageLoad(IMG img, VOID *v)
{
    if (!IMG_IsMainExecutable(img)) return;
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
                
                for (UINT32 w = 0; w < INS_MaxNumWRegs(ins); w++) {
                    REG r = REG_FullRegName(INS_RegW(ins, w));
                    if (r != REG_INVALID()) {
                        last_writer_pc[r] = INS_Address(ins);
                        last_writer_op[r] = INS_Opcode(ins);
                    }
                }


                // If the instructions is a conditional branch
                if (INS_Category(ins) == XED_CATEGORY_COND_BR)
                {
                    ADDRINT addr = INS_Address(ins);
                    br_in_rtn.push_back(addr); // Record this branch to label later
                    branch &b =br_info[addr];
                    b.opcode = OPCODE_StringShort(INS_Opcode(ins)); // Opcode of branch
                    if (INS_IsDirectControlFlow(ins))
                    {
                        b.offset = INS_DirectControlFlowTargetAddress(ins) - addr; // Branch's Offset 
                    }
                    b.size = INS_Size(ins); // Branch's instruction size 
                    int i = 0;

                    // Previous N Instructions from branch
                    for (INS prev = INS_Prev(ins); INS_Valid(prev) && i < NUM_PREV; prev = INS_Prev(prev))
                    {
                        b.prev_instr[i].opcode = INS_Opcode(prev); 
                        b.prev_instr[i].size = INS_Size(prev);
                        i = i + 1;
                    }
                    i = 0;
                    // Next N Instructions from branch
                    for (INS next = INS_Next(ins); INS_Valid(next) && i < NUM_PREV; next = INS_Next(next))
                    {
                        b.next_instr[i].opcode = INS_Opcode(next);
                        b.next_instr[i].size = INS_Size(next);
                        i = i + 1;
                    }


                    if (INS_Valid(flag_write))
                    {
                        b.flag_wr_instr.opcode = INS_Opcode(flag_write); // Opcode of the Flag_Write Instr
                        b.flag_wr_instr.pc = INS_Address(flag_write); // PC of Flag_Write Instr
                        b.flag_wr_instr.size = INS_Size(flag_write); // Size of Flag_Write Instr

                        for (UINT32 i = 0; i < INS_MaxNumRRegs(flag_write); i++)
                        {
                            REG r = INS_RegR(flag_write, i);
                            if (r == REG_INVALID()) continue;

                            regs new_reg;
                            new_reg.reg = r;
                            
                            // Check our "Single Pass" tracker instead of searching backwards
                            if (last_writer_pc.find(REG_FullRegName(r)) != last_writer_pc.end()) {
                                new_reg.define_instr_pc = last_writer_pc[REG_FullRegName(r)];
                                new_reg.define_instr_op = last_writer_op[REG_FullRegName(r)];
                            }
                            b.flag_wr_instr.reg_read.push_back(new_reg);
                        }

                        for (UINT32 i = 0; i < INS_MaxNumWRegs(flag_write); i++)
                        { 
                            REG r = INS_RegW(flag_write, i); // Register that Flag_write Instuction writes
                            regs new_reg;
                            new_reg.reg=r;
                            b.flag_wr_instr.reg_write.push_back(new_reg);
                        }
                    }
                    else
                    {
                        b.flag_wr_instr.pc = -1;
                    }
                    // Insert the BranchCount function before the instruction
                    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BranchCounter, IARG_FAST_ANALYSIS_CALL, IARG_PTR, &b ,IARG_END);
                    // Insesrt this fucntion TakenCounter to the taken flow of the instuctions
                    INS_InsertCall(ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)TakenCounter, IARG_FAST_ANALYSIS_CALL,IARG_PTR,&b, IARG_END);
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
    br_info.reserve(250000);
    bbls.reserve(1000000);
    PIN_AddFiniFunction(Fini, 0);
    TRACE_AddInstrumentFunction(Trace, 0);
    IMG_AddInstrumentFunction(ImageLoad, 0);
    PIN_StartProgram();
}
