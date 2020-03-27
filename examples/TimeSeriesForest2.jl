using Revise, CSV, DataFrames
using Tables

using MLJ
@load DecisionTreeClassifier
tree_model = DecisionTreeClassifier()

function RandomForestClassifierTS(X, y; n_trees::Int=200, min_interval::Int=3)
    transform_xt = InvFeatureGen(X, n_trees=n_trees, min_interval=min_interval)
    forest = Array{Machine{DecisionTreeClassifier},1}()
    for i in range(1, stop=n_trees)
        tree = machine(tree_model, transform_xt[i], y)
        mdl = deepcopy(tree)
        fit!(mdl)
        push!(forest, mdl)
    end
    forest
end

function InvFeatureGen(X; n_trees::Int=200, min_interval::Int=3)
    n_samps, series_length = size(X)
    transform_xt = Array{Array{Float64,2},1}()
    n_intervals = floor(Int, sqrt(series_length))
    intervals = zeros(Int, n_trees, n_intervals, 2)
    for i in range(1, stop = n_trees)
       transformed_x = Array{Float64,2}(undef, 3*n_intervals, n_samps)
       for j in range(1, stop = n_intervals)
           intervals[i,j,1] = rand(1:(series_length - min_interval))
           len = rand(1:(series_length - intervals[i,j,1]))
           if len < min_interval
               len = min_interval
           end
           intervals[i,j,2] = intervals[i,j,1] + len
           x = Array(1:len+1)
           Y = X[:, intervals[i,j,1]:intervals[i,j,2]]
           means = mean(Y, dims=2)
           stds =  std(Y, dims=2)
           slope = (mean(transpose(x).*Y, dims=2) -
                    mean(x)*mean(Y, dims=2)) / (mean(x.*x) - mean(x)^2)
           transformed_x[3*j-2,:] =  means
           transformed_x[3*j-1,:] =  stds
           transformed_x[3*j,:]   =  slope
       end
           push!(transform_xt, transpose(transformed_x))
    end
    return transform_xt
end

function proba_predict(forest, features)
    n_trees = length(forest)
    n_samps, _ = size(features[1])
    class_ = zeros(Float64, n_samps, 2)
    for i in range(1, stop=n_trees)
        for j in range(1, stop=n_samps)
            class_[j,1] = class_[j,1] + predict(forest[i], features[i])[j].prob_given_class[1]
        end
    end
    class_[:,2] = n_samps .- class_[:,1]
    class_
end


file  = "/Users/aa25desh/TimeSerise/GunPoint/GunPoint_TRAIN.ts"
df = DataFrame(CSV.File(file))
mat = Tables.matrix(df)
X = float.(mat[:, 1:149])
y = [Meta.parse(mat[i, 150]).args[3] for i in range(1, stop=49)]
y = CategoricalArray(y)

forest = RandomForestClassifierTS(X, y)

file1  = "/Users/aa25desh/TimeSerise/GunPoint/GunPoint_TEST.ts"
df1 = DataFrame(CSV.File(file1))
mat1 = Tables.matrix(df1)
X1 = float.(mat1[:, 1:149])
y1 = [Meta.parse(mat1[i, 150]).args[3] for i in range(1, stop=149)]

features = InvFeatureGen(X1)

print(proba_predict(forest, features))
