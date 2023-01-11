function get_subtrace(trace::Trace, addr)
    error("Not implemented.")
end

function get_subtrace(trace::Gen.DynamicDSLTrace, addr)
    return trace.trie[addr].subtrace_or_retval
end

function get_subtrace(trace::Gen.DynamicDSLTrace, addr::Pair)
    key, subaddr = addr
    return get_subtrace(get_subtrace(trace, key), addr)
end

function get_subtrace(trace::Gen.StaticIRTrace, addr)
    return Gen.static_get_subtrace(trace, addr)
end

function get_subtrace(trace::Gen.StaticIRTrace, addr::Pair)
    key, subaddr = addr
    return get_subtrace(get_subtrace(trace, key), addr)
end