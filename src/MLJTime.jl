__precompile__()

module MLJTime

using DecisionTree
using Statistics

export InvFeatureGen, RandomForestClassifierTS, predict_f, proba_predict

include("IntervalBasedForest.jl")

end #module
