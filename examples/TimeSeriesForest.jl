using Revise, CSV, DataFrames
using Tables

using DecisionTree
using Statistics

file  = "/Users/aa25desh/TimeSerise/GunPoint/GunPoint_TRAIN.ts"
df = DataFrame(CSV.File(file))
mat = Tables.matrix(df)
X = float.(mat[:, 1:149])
y = [Meta.parse(mat[i, 150]).args[3] for i in range(1, stop=49)]

function RandomForestClassifierTS(X, y; n_trees::Int=200, min_interval::Int=3)
    transform_xt = InvFeatureGen(X, n_trees=n_trees, min_interval=min_interval)
    model = DecisionTreeClassifier(pruning_purity_threshold=0.67)
    forest = Array{DecisionTreeClassifier,1}()
    for i in range(1, stop=n_trees)
        mdl = deepcopy(model)
        fit!(mdl, transform_xt[i], y)
        push!(forest, mdl)
    end
    forest
end

function InvFeatureGen(X; n_trees::Int=200, min_interval::Int=3)
    n_samps, series_length = size(X)
    transform_xt = Array{Array{Float64,2},1}()
    n_intervals = floor(Int, sqrt(series_length))
    intervals = zeros(Int, n_trees, 3*n_intervals, 2)
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

features = InvFeatureGen(X)
forest = RandomForestClassifierTS(X, y)

file1  = "/Users/aa25desh/TimeSerise/GunPoint/GunPoint_TEST.ts"
df1 = DataFrame(CSV.File(file1))
mat1 = Tables.matrix(df1)

X1 = float.(mat1[:, 1:149])
y1 = [Meta.parse(mat1[i, 150]).args[3] for i in range(1, stop=149)]

function predict_ff(forest, features)
    vv = zeros(Float64,2)
    for i in range(1, stop=200)
        vv += collect(proba_predict(forest,  (features[i])))
    end
    return vv
end

function proba_predict(forest, X)
    a = 0
    for tree in forest
        if predict(tree, X) == 1
            a += 1
        end
    end
    (a, length(forest)-a)
end

function InvFeatures(X, n_trees::Int=200, min_interval::Int=3)
    transform_xt = []
    series_length, = size(X)
    n_intervals = floor(Int, sqrt(series_length))
    intervals = zeros(Int, n_intervals, 2)
    for i in range(1, stop = n_trees)
        transformed_x = Array{Float64,1}(undef, 3*n_intervals)
        for j in range(1, stop = n_intervals)
            intervals[j,1] = rand(1:(series_length - min_interval))
            len = rand(1:(series_length - intervals[j,1]))
            if len < min_interval
                len = min_interval
            end
            intervals[j,2] = intervals[j,1] + len
            x = Array(1:len+1)
            Y = X[intervals[j,1]:intervals[j,2]]
            means = mean(Y)
            stds  = std(Y)
            slope = (mean(x.*Y) -
                     mean(x)*mean(Y)) / (mean(x.*x) - mean(x)^2)
            transformed_x[3*j-2] = means
            transformed_x[3*j-1] = stds
            transformed_x[3*j] = slope
        end
        push!(transform_xt, vcat(transformed_x))
    end
    return transform_xt
end

for i in range(1, stop=149)
    yy = InvFeatures(X1[i,:])
    a, b = predict_ff(forest, yy)
    c = a > b ? 1 : 2
    if y1[i] == c
        print("1")
    else
        print("0")
    end
end

#=
function predict_f(forest, X) #takes only single instance input of Type "transform_xt"
    a, b = proba_predict(forest, X)
    a > b ? 1 : 2
end
=#
