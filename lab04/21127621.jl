### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 0158408c-a77a-4a88-9d0b-211befe003f9
begin
	using  Distributions, Plots, LinearAlgebra, Random, Statistics
end

# ╔═╡ 40a07666-8f41-4138-8ae4-9f533d0a5057
md"""
# Lab04: Decision Tree and Naive Bayes

- Student ID: 21127621
- Student name: Âu Dương Khang
"""

# ╔═╡ ded99c73-a5fb-46ca-94e1-fd024842158f
md"""
**How to do your homework**


You will work directly on this notebook; the word `TODO` indicate the parts you need to do.

You can discuss ideas with classmates as well as finding information from the internet, book, etc...; but *this homework must be your*.

**How to submit your homework**

- Before submitting, save this file as `<ID>.jl`. For example, if your ID is 123456, then your file will be `123456.jl`. And export to PDF with name `123456.pdf` then submit zipped source code and pdf into `123456.zip` onto Moodle.

!!! danger
	**Note that you will get 0 point for the wrong submit**.

**Contents:**

- Decision Tree
- Naive Bayes
"""

# ╔═╡ 304dd910-9b9e-474b-8834-5f76083bdddb
md"""
### Import library
"""

# ╔═╡ fe221125-8865-4407-a7a0-ecc5fdf14d8d
Random.seed!(2022)

# ╔═╡ fcf859d3-bc67-4c5a-8a27-c6576259cdaa
md"""
### Load Iris dataset
"""

# ╔═╡ 69290c91-65a0-4db6-b032-20a5af96e2dd
# If you use Linux, use this function to download Iris dataset
function download_dataset(save_path::String="data")
	# setup directory
	mkpath(joinpath(dirname(@__FILE__), save_path))
	data_dir = joinpath(dirname(@__FILE__), save_path)
	download("https://archive.ics.uci.edu/static/public/53/iris.zip",joinpath(data_dir, "iris.zip"))
	iris_file = joinpath(data_dir, "iris.zip")
	cd(data_dir)
	run(`unzip $iris_file -d $data_dir`)
  	rm(iris_file)
	cd("..")
end

# ╔═╡ 2b05a3b2-83f6-441a-976a-34444c45c1df
# If you have downloaded the dataset yet, please uncomment this line below and run this cell. Otherwise, keep it in uncomment state.
# download_dataset()

# ╔═╡ 813061a7-80f4-4b71-853f-d505b91d39a5
# If you don't use Linux, I have no solution for you. Please makedir data, and goto https://archive.ics.uci.edu/dataset/53/iris for dowloading
# Then, you can extract data to this dir by yourself.

# Structure:
# ├── data
# │   ├── bezdekIris.data
# │   ├── Index
# │   ├── iris.data
# │   └── iris.names
# ├── Lab4.jl

# ╔═╡ 6f13aaa0-2897-4513-8bb4-1a1da993d7f4
function iris_dataloader(data_path::String="data/iris.data")
	# Initialize empty arrays to store data
	sepal_length = Float64[]
	sepal_width = Float64[]
	petal_length = Float64[]
	petal_width = Float64[]
	classes = Int64[]

	# open and read the data file
	open(data_path, "r") do file
		# read data each line
		for line in eachline(file)
			if line != ""
				parts = split(line, ",")
				push!(sepal_length, parse(Float64, parts[1]))
				push!(sepal_width, parse(Float64, parts[2]))
				push!(petal_length, parse(Float64, parts[3]))
				push!(petal_width, parse(Float64, parts[4]))
	
				if parts[5] == "Iris-setosa"
					push!(classes, 0)
				elseif parts[5] == "Iris-versicolor"
					push!(classes, 1)
				else
					push!(classes, 2)
				end
			end
		end
	end

	# concat features
	features = [sepal_length, sepal_width, petal_length, petal_width] 
	features = vcat(transpose.(features)...)
	return features, classes
end

# ╔═╡ 1b96253f-9f1f-4229-9933-f15932a52cc0
# function change_class_to_num(y)
# 	class = Dict("setosa"=> 0,"versicolor"=> 1, "virginica" => 2)
# 	classnums = [class[item] for item in y]
# 	return classnums
# end

# ╔═╡ bb38145f-7bf2-4528-bc7e-e089e2790830
function train_test_split(X, y, test_ratio=0.33)
	X = X'
	n = size(X)[1]
    idx = shuffle(1:n)
    train_size = 1 - test_ratio
    train_idx = view(idx, 1:floor(Int, train_size*n))
    test_idx = view(idx, (floor(Int, train_size*n)+1):n)

	X_train = X[train_idx,:]
	X_test = X[test_idx,:]
	
	y_train = y[train_idx]
	y_test = y[test_idx]
	
    return X_train, X_test, y_train, y_test
end

# ╔═╡ 08e37795-1799-475b-9ca6-149a174eef5d
begin
	# Load features, and labels for Iris dataset
	iris_features, iris_labels = iris_dataloader("data/iris.data")

	#split dataset into training data and testing data
	X_train, X_test, y_train, y_test = train_test_split(iris_features, iris_labels, 0.33)

	size(X_train), size(X_test), size(y_train), size(y_test)
end

# ╔═╡ 9f35e4bb-55df-4c3c-abcb-ce2696fe225a
md"""
## 1. Decision Tree: Iterative Dichotomiser 3 (ID3)
"""

# ╔═╡ ef69925f-9982-4c3a-8780-74c9f70fc070
md"""
### 1.1 Information Gain
"""

# ╔═╡ 36e71cbd-479d-42f6-a4cc-61f74bd94f27
md"""
Expected value of the self-information (entropy):
"""

# ╔═╡ 3e23510b-1ab6-4f85-8103-43f2e070d5bb
md"""
$$Entropy=-\sum_{i}^{n}p_ilog_{2}(p_i)$$
"""

# ╔═╡ 4e4c50ce-d1ca-400c-98e4-6ac26a82d7a3
md"""
The entropy function gets the smallest value if there is a value of $p_i$ equal to 1, reaches the maximum value if all $p_i$ are equal. These properties of the entropy function make it is an expression of the disorder, or randomness of a system, ...
"""

# ╔═╡ d3becd4a-4b54-4155-901d-af1bae7a39e2
"""
Parameters:
- `counts`: shape (n_classes): list number of samples in each class
- `n_samples:` number of data samples

Returns
- entropy 
"""
function entropy(counts, n_samples)
    #TODO
	entropy = 0
	for count in counts
		p_i = count / n_samples
		entropy -= p_i * log2(p_i)
	end
	
	return entropy

end

# ╔═╡ 7c7f868b-714f-4467-86a6-1df8041456f4
"""
Returns entropy of a divided group of data

Data may have multiple classes
"""
function entropy_of_one_division(division)
    
    n_samples = size(division, 1)
    n_classes = Set(division)
    
    counts=[]
	
    # count samples in each class then store it to list counts
    #TODO:
	counts = [count(x -> x == n_class, division) for n_class in n_classes]
	
    return entropy(counts,n_samples),n_samples

end

# ╔═╡ 975f01b5-a7c5-4321-9ad8-78b089f3fe61
"""
Returns entropy of a split
    
y_predict is the split decision by cutoff, True/Fasle
"""
function get_entropy(y_predict, y)
    n = size(y,1)
	# left hand side entropy
    entropy_true, n_true = entropy_of_one_division(y[y_predict])

	# right hand side entropy
    entropy_false, n_false = entropy_of_one_division(y[.~y_predict])
	
    # overall entropy
	 #TODO s=?
    s = entropy_true * n_true / n + entropy_false * n_false / n
    return s
end

# ╔═╡ 2a3a94d5-afe6-4fdc-8a73-a450570e8ebe
md"""
The information gain of classifying information set D by attribute A:

$$Gain(A)=Entrophy(D)-Entrophy_{A}(D)$$

At each node in ID3, an attribute is chosen if its information gain is highest compare to others.

All attributes of the Iris set are represented by continuous values. Therefore we need to represent them with discrete values. The simple way is to use a `cutoff` threshold to separate values of the data on each attribute into two part:` <cutoff` and `> = cutoff`.

To find the best `cutoff` for an attribute, we replace` cutoff` with its values then compute the entropy, best `cutoff` achieved when value of entropy is smallest  $\left (\arg \min Entrophy_ {A} (D) \right)$.
"""

# ╔═╡ 20cb6ddb-c984-4614-a4b3-f6742a733c80
md"""
### 1.2 Decision tree
"""

# ╔═╡ 300c4289-782b-4672-afa1-a16526941aa1
"""
Parameters:
- col_data: data samples in column
- y: label of training data

Returns
- minimum entropy, and cut-off value
"""
function find_best_split(col_data, y)
    min_entropy = 10
    cutoff = 0
	
    #Loop through col_data find cutoff where entropy is minimum
    
    for value in Set(col_data)
        y_predict = col_data .< value
        my_entropy = get_entropy(y_predict, y)
		
        #TODO
        #min entropy=?, cutoff=?
		if my_entropy <= min_entropy
            min_entropy = my_entropy
			cutoff = value
		end
    end
    return min_entropy, cutoff
end

# ╔═╡ 927a4b43-c1d3-4f93-a4da-aaaaadf4dca6
"""
Parameters:
- X: training data
- y: label of training data

Returns
- column index, cut-off value, and minimum entropy
"""
function find_best_split_of_all(X, y)
    col_idx = nothing
    min_entropy = 1
    cutoff = nothing

    for i in 1:size(X,2)
        col_data = X[:,i]
        entropy, cur_cutoff = find_best_split(col_data, y)
		
		# best entropy
        if entropy == 0                   
            return i, cur_cutoff, entropy
        elseif entropy <= min_entropy
            min_entropy = entropy
            col_idx = i
            cutoff = cur_cutoff
        end
    end
    return col_idx, cutoff, min_entropy
end

# ╔═╡ 4e274f58-ade7-4ac2-8b89-9c0f069c55ec
"""
Parameters:
- X: training data
- y: label of training data

Returns
- node 

node: each node represented by cutoff value and column index, value and children.
- cutoff value is thresold where you divide your attribute.
- column index is your data attribute index.
- value of node is mean value of label indexes, if a node is leaf all data samples will have same label.

Note that: we divide each attribute into 2 part => each node will have 2 children: left, right.
"""
function dtfit(X, y, node=Dict(), depth=0)
    #Stop conditions
    
    #if all value of y are the same 
    if all(y.==y[1])
        return Dict("val"=>y[1])
		
    else 
		# find one split given an information gain 
        col_idx, cutoff, entropy = find_best_split_of_all(X, y)  
		
        y_left = y[X[:,col_idx] .< cutoff]
        y_right = y[X[:,col_idx] .>= cutoff]
		
        node = Dict("index_col"=>col_idx,
                    "cutoff"=>cutoff,
                    "val"=> mean(y),
                    "left"=> Any,
                    "right"=> Any)
		
        left = dtfit(X[X[:,col_idx] .< cutoff, :], y_left, Dict(), depth+1)
        right= dtfit(X[X[:,col_idx] .>= cutoff, :], y_right, Dict(), depth+1)
		
        push!(node, "left" => left)
        push!(node, "right" => right)
		
        depth += 1 
    end
    return node
end

# ╔═╡ 2d9caa3a-2244-41b5-b286-2ca093652dea
function _dtpredict(tree, row)
    cur_layer = tree
    while haskey(cur_layer, "cutoff")
            if row[cur_layer["index_col"]] < cur_layer["cutoff"]
                cur_layer = cur_layer["left"]
            else
                cur_layer = cur_layer["right"]
            end
        end
    if !haskey(cur_layer, "cutoff")
        return get(cur_layer, "val", false)
    end
end

# ╔═╡ 0d015f12-64b7-453f-bfb8-b9f35b866b18
function dtpredict(tree, data)
    pred = []
    n_sample = size(data, 1)
    for i in 1:n_sample
        push!(pred, _dtpredict(tree, data[i,:]))
    end
    return pred
end

# ╔═╡ 8ef4dcff-016c-47db-83c3-1be7976408ec
md"""
### 1.3 Classification on Iris Dataset
"""

# ╔═╡ e5af1b95-8756-4b55-9365-fa06aaecce7b
function tpfptnfn_cal(y_test, y_pred, positive_class=1)
	true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

	# Calculate true positives, false positives, false negatives, and true negatives
    for (true_label, predicted_label) in zip(y_test, y_pred)
        if true_label == positive_class && predicted_label == positive_class
            true_positives += 1
		elseif true_label != positive_class && predicted_label == positive_class
            false_positives += 1
		elseif true_label == positive_class && predicted_label != positive_class
            false_negatives += 1
		elseif true_label != positive_class && predicted_label != positive_class
            true_negatives += 1
        end
    end

	return true_positives, false_positives, true_negatives, false_negatives
end

# ╔═╡ 843b6fe7-380a-42ea-878b-d65fa57433e3
tree = dtfit(X_train, y_train)

# ╔═╡ 0b1e782f-22ac-4c59-a1d3-ce0a32471d7c
begin
	pred = dtpredict(tree, X_test)

	acc = 0
	precision = 0
	recall = 0
	f1 = 0
	
	for i ∈ [0, 1, 2]
		# Calculate true positives, false positives, false negatives, and true negatives
	    true_positives, false_positives, true_negatives, false_negatives = tpfptnfn_cal(y_test, pred, i)
	
		# Calculate precision, recall, and F1-score
		acc += (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
	    precision += true_positives / (true_positives + false_positives)
	    recall += true_positives / (true_positives + false_negatives)
	end
	
	acc = acc / 3
	precision = precision / 3
	recall = recall / 3
	f1 = 2 * precision * recall / (precision + recall) 
	print(" acc: $acc\n precision: $precision\n recall: $recall\n f1_score: $f1\n")
end

# ╔═╡ 92dbc0c6-821b-4cb3-bf0a-521ca6d0d90e
md"""
## 2. Bayes Theorem

Bayes formulation
$$\begin{equation}
P\left(A|B\right)= \dfrac{P\left(B|A\right)P\left(A\right)}{P\left(B\right)}
\end{equation}$$

If $B$ is our data $\mathcal{D}$, $A$ and $w$ are parameters we need to estimate:

$$\begin{align}
    \underbrace{P(w|\mathcal{D})}_{Posterior}= \dfrac{1}{\underbrace{P(\mathcal{D})}_{Normalization}} \overbrace{P(\mathcal{D}|w)}^{\text{Likelihood}} \overbrace{P(w)}^{Prior}
    \end{align}$$
"""

# ╔═╡ a7e580b9-1802-426c-a1ce-7bee161b505b
md"""
#### Naive Bayes
To make it simple, it is often assumed that the components of the $D$ random variable (or the features of the $D$ data) are independent with each other, if $w$ is known. It mean:

$$P(\mathcal{D}|w)=\prod _{i=1}^{d}P(x_i|w)$$

- d: number of features

"""

# ╔═╡ dc2aefa1-d3e6-4582-9f69-759fae009db2
md"""
### 2.1. Probability Density Function
"""

# ╔═╡ 283c9682-465f-400b-ba3e-5be0c75a8a16
function maxHypo(hist)
    #find the hypothesis with maximum probability from hist
    #TODO
	max_value_index = argmax(collect(values(hist)))
    max_keys = collect(keys(hist))
    return max_keys[max_value_index]
end

# ╔═╡ d067dcda-8b13-4ef1-b28e-84e2f5b5422a
md"""
### 2.2 Classification on Iris Dataset
"""

# ╔═╡ b6b2f609-163a-4250-b74b-e2c0b940b1db
md"""
#### Gaussian Naive Bayes
"""

# ╔═╡ e65ebb4b-e825-4c4f-a2c2-b0ed3188a8b6
md"""
- Naive Bayes can be extended to use on continuous data, most commonly by using a normal distribution (Gaussian distribution).

- This extension called Gaussian Naive Bayes. Other functions can be used to estimate data distribution, but Gauss (or the normal distribution) is the easiest to work with since we only need to estimate the mean and standard deviation from the training data.
"""

# ╔═╡ a76dfc33-0e93-47ef-87d2-83f5eca81311
md"""
#### Define Gauss function
"""

# ╔═╡ f3135570-1a78-4926-bb4a-6aac475b2750
md"""
$$f\left(x;\mu,\sigma \right)= \dfrac{1}{\sigma \sqrt{2\pi}} 
\exp \left({-\dfrac{\left(x-\mu\right)^2}{2 \sigma^2}}\right)$$
"""

# ╔═╡ 8aca9054-e87c-4ba8-903d-0ce6b6aaaab5
function Gauss(std, mean, x)
    #Compute the Gaussian probability distribution function for x
    #TODO 
     return (1.0 / (std * sqrt(2 * π))) * exp(-((x - mean)^2) / (2 * std^2))
end

# ╔═╡ c203ca5a-4ab4-41fe-9d17-90a19bbd41ff
function likelihood(_mean=nothing, _std=nothing ,data=nothing, hypo=nothing)
    """
    Returns: res=P(data/hypo)
    -----------------
    Naive bayes:
        Atributes are assumed to be conditionally independent given the class value.
    """

    std=_std[hypo]
    mean=_mean[hypo]
    res=1
    #TODO
    #res=res*P(x1/hypo)*P(x2/hypo)...
    for i in 1:length(data)
        res *= Gauss(std[i], mean[i], data[i])
    end
    return res     
end

# ╔═╡ 588dc7c6-ffe6-4f95-bbbd-950ccd3e9f87
#update histogram for new data 
function update(_hist, _mean, _std, data)
    """
    P(hypo/data)=P(data/hypo)*P(hypo)*(1/P(data))
    """
    hist = copy(_hist)
    #P(hypo/data)=P(data/hypo)*P(hypo)*(1/P(data))
	
	#Likelihood * Prior
    #TODO
	s = 0
    for hypo in keys(hist)
		hypo_likelihood = hist[hypo] * likelihood(_mean, _std, data, hypo)
        hist[hypo] = hypo_likelihood 
		s+= hypo_likelihood
    end
    #Normalization
	
    #TODO: s=P(data)
    #s=?
    for hypo in keys(hist)
        hist[hypo] = hist[hypo]/s
    end
    return hist
end

# ╔═╡ 851b8060-77db-4dac-91d0-45b41e52f2af
function gfit(X, y, _std=nothing, _mean=nothing, _hist=nothing)
    """Parameters:
    X: training data
    y: labels of training data
    """
    n=size(X,1)
    #number of iris species
    #TODO
    #n_species=???
    n_species = length(Set(y))
    
   	hist_dict = Dict()
    mean_dict = Dict()
    std_dict = Dict()
    
    #separate  dataset into rows by class
    for hypo in Set(y)
        #rows have hypo label
        #TODO rows=
		rows = findall(a->a==hypo,y)
        #histogram for each hypo
        #TODO probability=?
		probability = length(rows) / n
        hist_dict[hypo] = probability
        
        #Each hypothesis represented by its mean and standard derivation
        """mean and standard derivation should be calculated for each column (or each attribute)"""
        #TODO mean[hypo]=?, std[hypo]=?
        mean_dict[hypo] = mean(X[rows, :], dims=1)
        std_dict[hypo] = std(X[rows, :], dims=1)
        
    end
    _mean = mean_dict
    _std = std_dict
    _hist = hist_dict
    return _hist, _mean, _std
end

# ╔═╡ 01284304-1dd9-42e2-abf7-a08537653bd3
function _gpredict(_hist, _mean, _std, data, plot=true)
    """
    Predict label for only 1 data sample
    ------------
    Parameters:
    data: data sample
    plot: True: draw histogram after update new record
    -----------
    return: label of data
    """
    hist = update(_hist, _mean, _std, data)
    if (plot == true)
        plt = bar(collect(keys(hist)), collect(values(hist)))
    end
    return maxHypo(hist)
end

# ╔═╡ d475b15a-7ac8-4f51-a828-7c5812d5835d
function plot_pdf(_hist)
     bar(collect(keys(_hist)), collect(values(_hist)))
end

# ╔═╡ 45e19116-2d2d-40fb-b432-ac99bc135151
function gpredict(_hist, _mean, _std, data)
    """Parameters:
    Data: test data
    ----------
    return labels of test data
    """
    pred=[]
    n_sample = size(data, 1)
    for i in 1:n_sample
        push!(pred, _gpredict(_hist, _mean, _std, data[i,:]))
    end
    return pred
end   

# ╔═╡ 96f39b4e-05bc-4b3b-bf0b-403769c80ed9
md"""
#### Show histogram of training data
"""

# ╔═╡ e6ea8368-a55f-4b90-bc22-5d64a7de9887
begin
	_hist, _mean, _std = gfit(X_train, y_train)
	plt = plot_pdf(_hist)
end

# ╔═╡ 3243a88e-3aa7-4e65-94f7-fb0a0df0aefd
md"""
#### Test wih 1 data record
"""

# ╔═╡ b9641fcb-813b-40ce-9e88-e8f096c6d895
begin
	#label of test_y[10]
	print("Label of X_test[10]: ", y_test[20])
	
	#update model and show histogram with X_test[10]:
	print("\nOur histogram after update X_test[10]: ", _gpredict(_hist, _mean, _std, X_test[20,:], true))
	
end

# ╔═╡ 1bda6fe2-80c6-42ce-9f0c-9b7202c1a764
md"""
#### Evaluate your Gaussian Naive Bayes model
"""

# ╔═╡ 850fde73-3e5f-4c1d-8413-7ec561cfcfdb
begin
	_pred = gpredict(_hist, _mean, _std, X_test)

	_acc = 0
	_p = 0
	_r = 0
	_f1 = 0
	
	#TODO: Self-define and calculate accuracy, precision, recall, and f1-score
	# Calculate accuracy, precision, recall, and f1-score
	for i in 1:10
	    _true_positives, _false_positives, _true_negatives, _false_negatives = tpfptnfn_cal(y_test, _pred)
	
	    _acc += (_true_positives + _true_negatives) / (_true_positives + _false_positives + _true_negatives + _false_negatives)
	    _p += _true_positives / (_true_positives + _false_positives)
	    _r += _true_positives / (_true_positives + _false_negatives)
	end
	_acc /= 10
	_p /= 10
	_r /= 10
	_f1 = 2 * _p * _r / (_p + _r)
	print(" acc: $_acc\n precision: $_p\n recall: $_r\n f1_score: $_f1\n")
end

# ╔═╡ f52d4dfd-d0de-4a0a-adb2-43e8067c22e2
md"""
**TODO**: F1, Recall and Precision report
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.102"
Plots = "~1.39.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "b66b8f8e3db5d7835fb8cbe2589ffd1cd456e491"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.17.0"

[[ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "3d5873f811f582873bb9871fc9c451784d5dc8c7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.102"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

[[JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d1b5f455bdd787aa7ac35d1f31f0bdb5d396ba27"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.27"

[[Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

[[PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "7c9196c8c83802d7b8ca7a6551a0236edd3bf731"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.0"

[[URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unitful]]
deps = ["ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

[[UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═40a07666-8f41-4138-8ae4-9f533d0a5057
# ╟─ded99c73-a5fb-46ca-94e1-fd024842158f
# ╟─304dd910-9b9e-474b-8834-5f76083bdddb
# ╠═0158408c-a77a-4a88-9d0b-211befe003f9
# ╠═fe221125-8865-4407-a7a0-ecc5fdf14d8d
# ╟─fcf859d3-bc67-4c5a-8a27-c6576259cdaa
# ╠═69290c91-65a0-4db6-b032-20a5af96e2dd
# ╠═2b05a3b2-83f6-441a-976a-34444c45c1df
# ╠═813061a7-80f4-4b71-853f-d505b91d39a5
# ╠═6f13aaa0-2897-4513-8bb4-1a1da993d7f4
# ╠═1b96253f-9f1f-4229-9933-f15932a52cc0
# ╠═bb38145f-7bf2-4528-bc7e-e089e2790830
# ╠═08e37795-1799-475b-9ca6-149a174eef5d
# ╟─9f35e4bb-55df-4c3c-abcb-ce2696fe225a
# ╟─ef69925f-9982-4c3a-8780-74c9f70fc070
# ╟─36e71cbd-479d-42f6-a4cc-61f74bd94f27
# ╟─3e23510b-1ab6-4f85-8103-43f2e070d5bb
# ╟─4e4c50ce-d1ca-400c-98e4-6ac26a82d7a3
# ╠═d3becd4a-4b54-4155-901d-af1bae7a39e2
# ╠═7c7f868b-714f-4467-86a6-1df8041456f4
# ╠═975f01b5-a7c5-4321-9ad8-78b089f3fe61
# ╠═2a3a94d5-afe6-4fdc-8a73-a450570e8ebe
# ╟─20cb6ddb-c984-4614-a4b3-f6742a733c80
# ╠═4e274f58-ade7-4ac2-8b89-9c0f069c55ec
# ╠═927a4b43-c1d3-4f93-a4da-aaaaadf4dca6
# ╠═300c4289-782b-4672-afa1-a16526941aa1
# ╠═0d015f12-64b7-453f-bfb8-b9f35b866b18
# ╠═2d9caa3a-2244-41b5-b286-2ca093652dea
# ╟─8ef4dcff-016c-47db-83c3-1be7976408ec
# ╠═e5af1b95-8756-4b55-9365-fa06aaecce7b
# ╠═843b6fe7-380a-42ea-878b-d65fa57433e3
# ╠═0b1e782f-22ac-4c59-a1d3-ce0a32471d7c
# ╟─92dbc0c6-821b-4cb3-bf0a-521ca6d0d90e
# ╟─a7e580b9-1802-426c-a1ce-7bee161b505b
# ╟─dc2aefa1-d3e6-4582-9f69-759fae009db2
# ╠═588dc7c6-ffe6-4f95-bbbd-950ccd3e9f87
# ╠═283c9682-465f-400b-ba3e-5be0c75a8a16
# ╟─d067dcda-8b13-4ef1-b28e-84e2f5b5422a
# ╟─b6b2f609-163a-4250-b74b-e2c0b940b1db
# ╟─e65ebb4b-e825-4c4f-a2c2-b0ed3188a8b6
# ╟─a76dfc33-0e93-47ef-87d2-83f5eca81311
# ╟─f3135570-1a78-4926-bb4a-6aac475b2750
# ╠═8aca9054-e87c-4ba8-903d-0ce6b6aaaab5
# ╠═c203ca5a-4ab4-41fe-9d17-90a19bbd41ff
# ╠═851b8060-77db-4dac-91d0-45b41e52f2af
# ╠═01284304-1dd9-42e2-abf7-a08537653bd3
# ╠═d475b15a-7ac8-4f51-a828-7c5812d5835d
# ╠═45e19116-2d2d-40fb-b432-ac99bc135151
# ╟─96f39b4e-05bc-4b3b-bf0b-403769c80ed9
# ╠═e6ea8368-a55f-4b90-bc22-5d64a7de9887
# ╟─3243a88e-3aa7-4e65-94f7-fb0a0df0aefd
# ╠═b9641fcb-813b-40ce-9e88-e8f096c6d895
# ╟─1bda6fe2-80c6-42ce-9f0c-9b7202c1a764
# ╠═850fde73-3e5f-4c1d-8413-7ec561cfcfdb
# ╟─f52d4dfd-d0de-4a0a-adb2-43e8067c22e2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
