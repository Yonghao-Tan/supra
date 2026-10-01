#include "tb/testcase_handoff_actions.hpp"
#include <iostream>

static void require(bool ok) { if (!ok) throw std::runtime_error("handoff self-test failed"); }

int main() {
    std::vector<forward_postprocess_next_token_descriptor> boundary_rows(395);
    for (unsigned p = 0; p < boundary_rows.size(); ++p) {
        auto& row = boundary_rows[p];
        row.token_position = row.kv_index = row.source_index = p;
        row.activation_bits = 8;
    }
    require(testcase_handoff::metadata(boundary_rows, 395, 0).size() == 7136);
    for (unsigned p = 220; p < 235; ++p) boundary_rows[p].activation_bits = 4;
    require(testcase_handoff::metadata(boundary_rows, 395, 0).size() == 7168);
    std::map<std::uint64_t, std::uint8_t> memory;
    const auto read = [&](std::uint64_t address) { return int(memory[address]); };
    const auto write = [&](std::uint64_t address, std::uint8_t value) { memory[address] = value; return 0; };
    const auto set = [&](std::uint64_t address, unsigned width, std::uint64_t value) {
        for (unsigned b = 0; b < width; ++b) memory[address + b] = value >> (8 * b);
    };
    const auto get = [&](std::uint64_t address, unsigned width) {
        std::uint64_t value = 0;
        for (unsigned b = 0; b < width; ++b) value |= std::uint64_t(memory[address + b]) << (8 * b);
        return value;
    };
    constexpr unsigned config = 0x1000, metadata = 0x2000, states = 0x10000, table = 0x12000;
    for (unsigned i = 0; i < 64; ++i) {
        set(states + i * 32, 2, i);
        set(states + i * 32 + 5, 1, i < 32 ? FORWARD_POSTPROCESS_LOCKED : FORWARD_POSTPROCESS_MASKED);
        set(states + i * 32 + 7, 1, i < 32 ? 8 : 4);
        set(states + i * 32 + 8, 4, (i + 3) % 64);
        set(states + i * 32 + 20, 4, 7);
    }
    set(states + 25, 1, 1);
    nlohmann::json action{{"kind", "state_to_cross_block_inputs"}, {"config_address", config},
        {"metadata_address", metadata}, {"metadata_capacity", 4096}, {"state_address", states},
        {"sequence", 64}, {"current_begin", 32}, {"capture", 7}, {"table_address", table},
        {"probability_address", 0x20000}, {"jobs_address", 0x100000}, {"jobs_capacity", 256},
        {"refresh_address", config + 320}, {"output_address", 0x200000}};
    auto result = testcase_handoff_actions(nlohmann::json{{"handoff_actions", {action}}}, read, write);
    require(get(config + EXECUTION_CONFIG_TOTAL_TOKEN_COUNT_OFFSET, 2) == 64);
    require(result[0]["changed"] == 1 && result[0]["relation_jobs"] == 32);
    unsigned bytes = 0;
    auto tokens = testcase_handoff::read_metadata(get, metadata, 4096, bytes);
    require(tokens.size() == 64 && tokens.back().round == 1);
    for (const auto& token : tokens) {
        require(get(token.address, 4) == (token.position + 3) % 64);
        require(token.bits == (token.position < 32 ? 8u : 4u));
    }
    // Actual precision changes must rebuild both physical packing and the inverse map.
    for (unsigned i = 32; i < 64; ++i) set(states + i * 32 + 7, 1, 8);
    testcase_handoff_actions(nlohmann::json{{"handoff_actions", {action}}}, read, write);
    tokens = testcase_handoff::read_metadata(get, metadata, 4096, bytes);
    for (const auto& token : tokens) require(token.bits == 8);
    auto damaged = memory;
    set(metadata + 32 + 32 * 16, 1, 255);
    bool rejected = false;
    try { testcase_handoff::read_metadata(get, metadata, 4096, bytes); }
    catch (const std::exception&) { rejected = true; }
    require(rejected); memory = damaged;
    std::vector<forward_postprocess_next_token_descriptor> deep(2);
    for (unsigned i = 0; i < 2; ++i) {
        deep[i].source_index = deep[i].token_position = deep[i].kv_index = 62 + i;
        deep[i].activation_bits = 8;
    }
    const auto packed = testcase_handoff::metadata(deep, 64, 7);
    for (unsigned i = 0; i < 4096; ++i) memory[metadata + i] = i < packed.size() ? packed[i] : 0;
    action = {{"kind", "execution_from_metadata"}, {"config_address", config}, {"metadata_address", metadata},
        {"metadata_capacity", 4096}, {"source_index_offset", 64}, {"output_address", 0x200000},
        {"output_capacity_tokens", 128}, {"commit_table_address", table}, {"sequence", 64}};
    testcase_handoff_actions(nlohmann::json{{"handoff_actions", {action}}}, read, write);
    tokens = testcase_handoff::read_metadata(get, metadata, 4096, bytes);
    require(get(tokens[0].address, 4) == 126 && get(tokens[1].address, 4) == 127);
    require(get(config + EXECUTION_CONFIG_OUTPUT_HIDDEN_LIMIT_OFFSET, 8) == 0x200000 + 128 * 8192);
    require(get(table + 62 * 8, 8) == 256 && get(table, 8) == 0);
    action["metadata_capacity"] = 32;
    rejected = false;
    try { testcase_handoff_actions(nlohmann::json{{"handoff_actions", {action}}}, read, write); }
    catch (const std::exception&) { rejected = true; }
    require(rejected);
    // Reuse a fixed DDR layout with live token IDs and live current/future state.
    memory.clear();
    constexpr unsigned layout = 0x4000, predictions = 0x6000, result_address = 0x8000;
    std::vector<forward_postprocess_next_token_descriptor> selected(2);
    selected[0] = {501, 33, 33, FORWARD_POSTPROCESS_EMBEDDING_TOKEN, 8, 0, 0};
    selected[1] = {902, 65, 65, FORWARD_POSTPROCESS_EMBEDDING_TOKEN, 8, 0, 0};
    const auto actual_metadata = testcase_handoff::metadata(selected, 96, 12);
    auto fixed_metadata = actual_metadata;
    fixed_metadata[3] |= 2; fixed_metadata[30] = 2; fixed_metadata[31] = 1;
    for (unsigned b = 0; b < actual_metadata.size(); ++b) {
        memory[metadata+b] = actual_metadata[b]; memory[layout+b] = fixed_metadata[b];
    }
    set(layout+32, 4, 0); set(layout+48, 4, 1);
    set(metadata+24, 6, 2); // Actual Source A row suppresses its KV write.
    set(result_address, 4, 2);
    for (unsigned i=0; i<64; ++i) {
        set(states+i*32, 2, 32+i);
        set(states+i*32+5, 1, i==1 ? FORWARD_POSTPROCESS_MASKED : FORWARD_POSTPROCESS_LOCKED);
        set(states+i*32+8, 4, i==1 ? 501 : 902);
        set(states+i*32+19, 1, 1); set(states+i*32+28, 3, 0x013f80);
    }
    set(predictions,2,33); set(predictions+32,2,65);
    set(config+EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET,2,2);
    set(config+EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET,8,predictions);
    action = {{"kind","execution_from_metadata"},{"config_address",config},
        {"metadata_address",metadata},{"metadata_capacity",4096},
        {"layout_address",layout},{"layout_capacity",fixed_metadata.size()},
        {"current_begin",32},{"current_end",64},{"future_result_address",result_address},
        {"state_address",states},{"state_count",64}};
    testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(get(layout+32,4)==501 && get(layout+48,4)==902 && get(layout+24,6)==2);
    require(get(states+32+7,1)==8 && get(states+32+19,1)==0);
    require(get(states+33*32+19,1)==1 && get(states+33*32+28,3)==0x013f80);
    require(get(states+32+28,3)==0 && get(predictions+8,4)==501 && get(predictions+32+20,2)==1);
    require(get(config+EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET,8)==layout);
    for (unsigned embedding_mask : {1u, 2u, 0u, 3u}) {
        for (unsigned row = 0; row < 2; ++row)
            set(metadata + 32 + row * 16 + 9, 1, embedding_mask & (1u << row) ?
                FORWARD_POSTPROCESS_EMBEDDING_TOKEN : FORWARD_POSTPROCESS_RESIDENT_HIDDEN);
        testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
        require((get(layout + 3, 1) & 1) == unsigned(embedding_mask != 0));
    }
    // A separate L31 layout gathers hidden rows by their normal packed ordinals.
    constexpr unsigned directory = 0x3000, l31_layout = 0x5000;
    for (unsigned b=0;b<fixed_metadata.size();++b) memory[l31_layout+b] = fixed_metadata[b];
    set(layout+3,1,1);
    set(config+EXECUTION_CONFIG_FLAGS_OFFSET,4,1u<<13);
    set(config+EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET,8,directory);
    set(config+EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET,8,l31_layout+fixed_metadata.size());
    set(directory,8,layout); set(directory+8,8,layout+fixed_metadata.size());
    set(directory+16,8,l31_layout); set(directory+24,8,l31_layout+fixed_metadata.size());
    testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(get(config+EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET,8)==directory);
    require(get(layout+32,4)==501 && get(layout+48,4)==902);
    require(get(l31_layout+32,4)==0 && get(l31_layout+48,4)==1);
    require(get(l31_layout+41,1)==FORWARD_POSTPROCESS_RESIDENT_HIDDEN);
    require((get(l31_layout+3,1) & 1) == 0);
    require(get(l31_layout+24,6)==2 && get(predictions+32+20,2)==1);
    set(metadata+32+4,2,34);
    rejected=false;
    try { testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write); }
    catch (const std::exception&) { rejected=true; }
    require(rejected);
    // Baseline has no selector output. Build every A8 embedding source from
    // the persistent token table and the actual completed current block.
    memory.clear();
    constexpr unsigned next_states = 0x14000;
    std::vector<forward_postprocess_next_token_descriptor> full(96);
    for (unsigned p = 0; p < full.size(); ++p) {
        set(table+p*4,4,100+p);
        full[p] = {999, static_cast<std::uint16_t>(p), static_cast<std::uint16_t>(p),
                   FORWARD_POSTPROCESS_RESIDENT_HIDDEN, 8, 0, 0};
    }
    const auto full_layout = testcase_handoff::metadata(full,96,5);
    for (unsigned b=0;b<full_layout.size();++b) memory[layout+b] = full_layout[b];
    memory[layout+full_layout.size()] = 49; // Adjacent data is outside the layout capacity.
    const auto prepare_state = [&](unsigned begin, unsigned capture, bool reset, bool complete) {
        unsigned predictions_count=0;
        for (unsigned row=0;row<32;++row) {
            const auto address = reset ? next_states+row*32 : states+row*32;
            set(address,2,begin+row);
            set(address+5,1,complete || (!reset && row==0) ? FORWARD_POSTPROCESS_LOCKED : FORWARD_POSTPROCESS_MASKED);
            set(address+7,1,8); set(address+8,4,reset ? get(table+(begin+row)*4,4) : 400+row);
            set(address+12,4,UINT32_MAX); set(address+20,4,capture);
            if (!reset) for (unsigned b=0;b<32;++b) memory[next_states+row*32+b]=memory[address+b];
            if (get(next_states+row*32+5,1)==FORWARD_POSTPROCESS_MASKED)
                set(predictions+predictions_count++*32,2,begin+row);
        }
        set(config+EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET,2,predictions_count);
    };
    prepare_state(32,5,false,false);
    set(config+EXECUTION_CONFIG_SEQUENCE_LENGTH_OFFSET,2,96);
    set(config+EXECUTION_CONFIG_PREDICTION_TABLE_BASE_OFFSET,8,predictions);
    action = {{"kind","full_sequence_from_state"},{"config_address",config},
        {"metadata_address",metadata},{"metadata_capacity",4096},
        {"layout_address",layout},{"layout_capacity",full_layout.size()},
        {"sequence",96},{"capture",5},{"source_current_begin",32},{"current_begin",32},{"current_end",64},
        {"source_state_address",states},{"table_address",table},
        {"state_address",next_states},{"state_count",32},{"reset_current_state",false}};
    testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    tokens=testcase_handoff::read_metadata(get,layout,full_layout.size(),bytes);
    for (const auto& token : tokens) {
        const unsigned wanted = token.position>=32 && token.position<64 ? 400+token.position-32 : 100+token.position;
        require(token.bits==8 && get(token.address,4)==wanted && get(table+token.position*4,4)==wanted);
        require(get(token.address+9,1)==FORWARD_POSTPROCESS_EMBEDDING_TOKEN);
    }
    require(get(predictions+8,4)==401 && get(predictions+20,2)==33);
    // Same block, later actual update. Only the new live token may change.
    prepare_state(32,6,false,false); set(states+32+8,4,777); set(next_states+32+8,4,777);
    action["capture"]=6;
    testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(get(table+33*4,4)==777 && get(table+32*4,4)==400 && get(table+4,4)==101);
    // Completion advances to the next block without losing prior token IDs.
    prepare_state(32,7,false,true); set(states+32+8,4,777);
    prepare_state(64,7,true,false);
    action["capture"]=7; action["current_begin"]=64; action["current_end"]=96;
    action["reset_current_state"]=true;
    testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(get(table+33*4,4)==777 && get(table+64*4,4)==164 && get(predictions+8,4)==164);
    const auto valid=memory;
    set(states+20,4,6);
    rejected=false;
    try { testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write); }
    catch (const std::exception&) { rejected=true; }
    require(rejected); memory=valid;
    set(layout+32+4,2,95); // Duplicated/missing position in the fixed consumer layout.
    rejected=false;
    try { testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write); }
    catch (const std::exception&) { rejected=true; }
    require(rejected);
    memory=valid;
    set(next_states+5,1,FORWARD_POSTPROCESS_TENTATIVE);
    set(config+EXECUTION_CONFIG_PREDICTION_COUNT_OFFSET,2,1);
    set(predictions,2,64);
    action={{"kind","execution_from_metadata"},{"config_address",config},
        {"metadata_address",layout},{"metadata_capacity",full_layout.size()},
        {"layout_address",layout},{"layout_capacity",full_layout.size()},
        {"current_begin",64},{"current_end",96},{"state_address",next_states},
        {"state_count",32},{"closeout_kind",1}};
    testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(get(predictions+7,1)==FORWARD_POSTPROCESS_TENTATIVE);
    require(get(next_states+32+5,1)==FORWARD_POSTPROCESS_MASKED);
    memory.clear();
    constexpr unsigned history=0x14000, event=0x15000, selector=0x16000;
    constexpr unsigned completed_meta=0x18000, boundary_layout=0x1a000, boundary_next=0x1c000;
    constexpr unsigned sequence=129, previous=33, current=65;
    set(history,2,previous); set(history+2,2,sequence); set(history+4,4,10);
    set(history+8,4,1u<<2); set(history+16+5,1,1);
    set(event,4,11); set(event+16,4,1u<<7); set(event+8,4,1u<<9);
    std::vector<forward_postprocess_next_token_descriptor> whole(sequence), completed(64);
    for (unsigned p=0;p<sequence;++p) {
        set(table+p*8,8,std::uint64_t(9000+p)<<35);
        set(selector+p*8,8,(std::uint64_t(p)<<35)|(std::uint64_t(p)<<23)|
            (std::uint64_t(p>=current && p<current+32)<<8));
        whole[p].source_index=9000+p; whole[p].source=FORWARD_POSTPROCESS_EMBEDDING_TOKEN;
        whole[p].token_position=whole[p].kv_index=p; whole[p].activation_bits=p==current+1?8:4;
    }
    for (unsigned row=0;row<64;++row) {
        const unsigned p=previous+row, address=states+row*32;
        set(address,2,p); set(address+2,1,row%32); set(address+3,1,row/32); set(address+4,1,1+row/32);
        set(address+5,1,row<32?FORWARD_POSTPROCESS_LOCKED:row==33?FORWARD_POSTPROCESS_TENTATIVE:FORWARD_POSTPROCESS_MASKED);
        set(address+7,1,4); set(address+8,4,50000+p); set(address+12,4,60000+p);
        set(address+16,2,row<32?4:65535); set(address+20,4,11);
        set(address+19,1,row==3||row==33); set(address+31,1,row==33);
        completed[row].source_index=p; completed[row].token_position=completed[row].kv_index=p;
        completed[row].activation_bits=4;
    }
    const auto completed_bytes=testcase_handoff::metadata(completed,sequence,10);
    const auto whole_bytes=testcase_handoff::metadata(whole,sequence,11);
    for (unsigned b=0;b<completed_bytes.size();++b) memory[completed_meta+b]=completed_bytes[b];
    for (unsigned b=0;b<whole_bytes.size();++b) memory[boundary_layout+b]=whole_bytes[b];
    set(config+EXECUTION_CONFIG_TOKEN_METADATA_BASE_OFFSET,8,boundary_layout);
    set(config+EXECUTION_CONFIG_TOKEN_METADATA_LIMIT_OFFSET,8,boundary_layout+whole_bytes.size());
    action={{"kind","packed_boundary_from_state"},{"config_address",config},
        {"metadata_address",metadata},{"metadata_capacity",4096},{"sequence",sequence},{"capture",11},
        {"current_begin",current},{"current_end",current+32},{"source_current_begin",previous},
        {"source_state_address",states},{"source_state_count",64},{"source_event_address",event},
        {"source_metadata_address",completed_meta},{"source_metadata_capacity",completed_bytes.size()},
        {"history_address",history},{"token_table_address",table},{"table_address",selector},
        {"layout_address",boundary_layout},{"layout_capacity",whole_bytes.size()},
        {"precision_policy","original"},{"maturity_age",3},{"block_initialization_all_a8",false},{"context_bits",4},
        {"historical_only",false},{"block_index",2},{"next_state_address",boundary_next},{"next_state_count",64},
        {"probability_address",0x20000},{"probability_limit",0x200000},
        {"jobs_address",0x200000},{"jobs_capacity",4096},
        {"probability_config_address",config+496},{"refresh_address",config+320}};
    const auto initial_boundary=memory;
    result=testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(result[0]["transition_mask"]==((1u<<2)|(1u<<7)|(1u<<9)));
    require(get(history,2)==current && get(history+8,4)==0 && get(history+4,4)==11);
    require(get(table+5*8,8)>>35==9005 && get(table+36*8,8)>>35==50036);
    require((get(selector+5*8,8)&(1u<<10)) && (get(selector+36*8,8)&(1u<<10)));
    require(!(get(selector+66*8,8)&(1u<<10)));
    require(get(boundary_next+32+5,1)==FORWARD_POSTPROCESS_TENTATIVE && get(boundary_next+32+31,1)==0);
    require(get(boundary_next+32+12,4)==60066 && get(boundary_next+32+16,2)==65535);
    require(get(boundary_next+32*32+12,4)==UINT32_MAX && get(boundary_next+32*32+16,2)==65535);
    tokens=testcase_handoff::read_metadata(get,boundary_layout,whole_bytes.size(),bytes);
    for (const auto& row:tokens) {
        require(get(row.address,4)==(row.position>=previous && row.position<previous+64?50000+row.position:9000+row.position));
        require(row.bits==(row.position==current+1?8u:4u));
    }
    require(get(config+496+32,8)==((1u<<(35/8))|(1u<<(40/8))|(1u<<(42/8))));
    // An unchanged-token admission is in the transition union. An empty union
    // instead generates current-to-context jobs, without inventing changes.
    memory=initial_boundary; set(history+8,4,0); set(event+8,4,0); set(event+16,4,0);
    result=testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write);
    require(result[0]["transition_mask"]==0 && get(config+320+164,4)>0);
    memory=initial_boundary; set(event,4,9);
    rejected=false;
    try { testcase_handoff_actions(nlohmann::json{{"handoff_actions",{action}}},read,write); }
    catch (const std::exception&) { rejected=true; }
    require(rejected);
    std::cout << "PASS handoff actual tokens, baseline steps, packed boundary history/state, empty transitions and layout checks\n";
}
