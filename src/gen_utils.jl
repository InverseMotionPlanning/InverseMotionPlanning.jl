import GenParticleFilters:
    ParticleFilterView, get_log_norm_weights, get_norm_weights

"Sample from the standard Gumbel distribution."
function randgumbel()
    return -log(-log(rand()))
end

"Sample from a discrete Boltzmann distribution given unnormalized log probabilities."
function randboltzmann(elements, log_weights)
    chosen, chosen_weight = nothing, -Inf
    # Gumbel-max reservoir sampling
    for (elem, weight) in zip(elements, log_weights) 
        weight += randgumbel()
        if weight > chosen_weight
            chosen = elem
            chosen_weight = weight
        end
    end
    return chosen
end

function Gen.get_log_weights(state::ParticleFilterView, replace_nan::Bool)
    return replace_nan ?
        replace(state.log_weights, NaN => -Inf) : state.log_weights
end

function get_log_norm_weights(state::ParticleFilterView, replace_nan::Bool)
    return GenParticleFilters.lognorm(get_log_weights(state, replace_nan))
end

function get_norm_weights(state::ParticleFilterView, replace_nan::Bool)
    return exp.(get_log_norm_weights(state, replace_nan))
end

"""
    get_norm_weights(state::ParticleFilterState)

Return the vector of normalized weights for the current state,
one for each particle.
"""
get_norm_weights(state::ParticleFilterView) = exp.(get_log_norm_weights(state))


"Given a hierarchical trace, return the subtrace located at `addr`."
function get_subtrace(trace::Trace, addr)
    error("Not implemented.")
end

function get_subtrace(trace::Trace, addr::Nothing)
    return trace
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

"""
Given a `selection`, decompose a `trace` into an iterable over corresponding 
addresses, subtraces, and subselections.
"""
function subtrace_selections(
    trace::Trace, selection::Selection, subtrace_type::Type{<:Trace}
)   
    base_iter = selection isa HierarchicalSelection ?
        get_subselections(selection) : 
        get_submaps_shallow(get_selected(get_choices(trace), selection))
    map_iter = Iterators.map(base_iter) do (key, submap_or_selection)
        subtrace = get_subtrace(trace, key)
        subselection = selection isa HierarchicalSelection ?
            submap_or_selection : selection
        sub_iter = subtrace_selections(subtrace, subselection, subtrace_type)
        sub_iter = Iterators.map(sub_iter) do (addr, tr, sel)
            addr = isnothing(addr) ? key : key => addr
            return (addr, tr, sel)
        end
        return sub_iter
    end
    return Iterators.flatten(map_iter)
end

function subtrace_selections(
    trace::T, selection::Selection, subtrace_type::Type{T}
) where {T <: Trace}
    return ((nothing, trace, selection),)
end
