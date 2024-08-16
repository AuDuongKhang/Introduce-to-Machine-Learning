### A Pluto.jl notebook ###
# v0.19.9

#> [frontmatter]
#> title = "Lab 05: Support-Vector Networks"
#> date = "2023-12-02"
#> tags = ["Machine Learning", "Statistical Learning Theory", "Classification", "Intro2ML ", "Lab5 "]
#> description = "Implement primal/ kernel SVM"
#> license = "Copyright © Dept. of CS, VNUHCM-University of Science, 2023. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License"

using Markdown
using InteractiveUtils

# ╔═╡ 287aba39-66d9-4ff6-9605-1bca094a1ce5
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI # visualization purpose
	TableOfContents(title="📚 Table of Contents", indent=true, depth=3, aside=true)
end
  ╠═╡ =#

# ╔═╡ 08c5b8cf-be45-4d11-bd24-204b364a8278
using Plots, Distributions, LinearAlgebra, Random

# ╔═╡ 86f24f32-d8ee-49f2-b71a-bd1c5cd7a28f
# edit the code below to set your name and student identity number (i.e. the number without @student.hcmus.edu.vn)

student = (name = "Âu Dương Khang", id = "21127621")

# you might need to wait until all other cells in this notebook have completed running.
# scroll around the page to see what's up

# ╔═╡ eeae9e2c-aaf8-11ed-32f5-934bc393b3e6
md"""
Submission by: **_$(student.name)_** ($(student.id)@student.hcmus.edu.vn)
"""

# ╔═╡ cab7c88c-fe3f-40c4-beea-eea924d67975
md"""
# **Homework 5**: Support Vector Networks
`CSC14005`, Introduction to Machine Learning

This notebook was built for FIT@HCMUS student to learn about Support Vector Machines/or Support Vector Networks in the course CSC14005 - Introduction to Machine Learning. 

## Instructions for homework and submission

It's important to keep in mind that the teaching assistants will use a grading support application, so you must strictly adhere to the guidelines outlined in the instructions. If you are unsure, please ask the teaching assistants or the lab instructors as soon as you can. **Do not follow your personal preferences at stochastically**

### Instructions for doing homework

- You will work directly on this notebook; the word **TODO** indicates the parts you need to do.
- You can discuss the ideas as well as refer to the documents, but *the code and work must be yours*.

### Instructions for submission

- Before submitting, save this file as `<ID>.jl`. For example, if your ID is 123456, then your file will be `123456.jl`. Submit that file on Moodle.
	
!!! danger
	**Note that you will get 0 point for the wrong submit**.

### Content of the assignment

- Recall: Perceptron & Geometriy Margin
- Linear support vector machine (Hard-margin, soft-margin)
- Popular non-linear kernels
- Computing SVM: Primal, Dual
- Multi-class SVM

### Others

Other advice for you includes:
- Starting early and not waiting until the last minute
- Proceed with caution and gentleness. 

"Living 'Slow' just means doing everything at the right speed – quickly, slowly, or at whatever pace delivers the best results." Carl Honoré.

- Avoid sources of interference, such as social networks, games, etc.

"""

# ╔═╡ 99329d11-e709-48f0-96b5-32ae0cac1f50
Random.seed!(0)

# ╔═╡ bbccfa2d-f5b6-49c7-b11e-53e419808c1b
html"""
<p align="center">
  <img src="https://lnhutnam.github.io/resources/images/yinyang.png" />
</p>
"""

# ╔═╡ 4321f49b-1057-46bc-8d67-be3122be7a68
md"""
## Problem statement

Let $\mathcal{D} = \{(x_i, y_i) | x_i \in \mathbb{R}^{d}, y_i \in \{-1, 1\}\}_{i=1}^{n}$ be a dataset which is a set of pairs where $x_i \in \mathbb{R}^d$ is *data point* in some $d$-dimension vector space, and $y_i \in \{-1, 1\}$ is a *label* of the corespondent $x_i$ data point classifying it to one of the two classes.

The model is trained on $\mathcal{D}$ after which it is present with $x_{i+1}$, and is asked to predict the label of this previously unseen data point.

The prediction function is donated by $f(x) : \mathbb{R}^d \rightarrow \{-1, 1\}$
"""

# ╔═╡ 4fdaeeda-beee-41e1-a5f0-3209151a880d
md"""
## Recall: Perceptron & Geometry Margin (Maximum 2.5 points)

In fact, it is always possible to come up with such a "perfect" binary function if training samples are distinct. However, it is unclear whether such rules are applicable to data that does not exist in the training set. We don't need "learn-by-heart" learners; we need "intelligent" learners. More especially, such trivial rules do not suffice because our task is not to correctly classify the training set. Our task is to find a rule that works well for all new samples we would encounter in the access control setting; the training set is merely a helpful source of information to find such a function. We would like to find a classifier that "generalizes" well.

The key to finding a generalized classifier is to constrain the set of possible binary functions we can entertain. In other words, we would like to find a class of classifier functions such that if a function in this class works well on the training set, it is also likely to work well on the unseen images. This problem is considered a key problem named "model selection" in machine learning.
"""

# ╔═╡ ec906d94-8aed-4df1-932c-fa2263e6325d
md"""
### Linear classifiers through origin

For simplicity, we will just fix the function class for now. We will only consider a type of *linear classifiers*. For more formally, we consider the function of the form:

$f(\mathbf{x}, \theta) = \text{sign}(\theta_1\mathbf{x}_1 + \theta_2\mathbf{x}_2 + \dots + \theta_d\mathbf{x}_d) = \text{sign}(\theta^\top\mathbf{x})$
where $\theta = [\theta_1, \theta_2, \dots, \theta_d]^\top$ is a column vector of real valued parameters.

Different settings of the parameters give different functions in this class, i.e., functions whose value or output in $\{-1, 1\}$ could be different for some input $\mathbf{x}$.
"""

# ╔═╡ 42882ca3-8df6-4fb8-8884-29337373bac5
md"""
### Perceptron Learning Algorithms

After chosen a class of functions, we still have to find a specific function in this class that works well on the training set. This task often refers to estimation problem in machine learning. We would like to find $\theta$ that minimize the *training error*, i.e we would like to find a linear classifier that make fewest mistake in the training set.

$\mathcal{L}(\theta) = \frac{1}{n}\sum_{t=1}^n\left(1-\delta(y_t, f(\mathbf{x}; \theta))\right) = \frac{1}{n}\sum_{t=1}^n\text{Loss}(y_t, f(\mathbf{x}; \theta))$
where $\delta(y, y') = 1$ if $y=y'$ and $0$ if otherwise.

Perceptron update rule: Let $k$ donates the number of parameter updates we have performed and $\theta^{(k)}$ is the parameter vector after $k$ updates. Initially $k=0$, and $\theta^{(k)} = 0$. We the loop through all the training instances $(\mathbf{x}_t, y)t)$, and updates the parameters only in response to mistakes,

$$\begin{cases}
\theta^{(k+1)} \leftarrow \theta^{(k)} + y_t\mathbf{x}_t \text{ if } y_t(\theta^{(k+1)})^\top\mathbf{x}_t < 0 \\
\text{The parameters unchanged}\end{cases}$$

![Geometry intuition of Perceptron](https://lnhutnam.github.io/assets/images_posts/pla/linear_classfier.png)
"""

# ╔═╡ e78094ff-6565-4e9d-812e-3a36f78731ed
begin
	n = 1000 # sample size
	d = 2; # dimensionality of data
	μ = 5 # mean
	Σ = 8 # variance
end

# ╔═╡ d0ecb21c-6189-4b10-9162-1b94424f49ce
points1ₜᵣₐᵢₙ = rand(MvNormal([Σ, μ], 5 .* [2 (μ - d)/Σ; (μ - d)/Σ d]), n ÷ 2)

# ╔═╡ 921e8d15-e751-4976-bb80-2cc09e6c950e
points2ₜᵣₐᵢₙ = rand(MvNormal([-μ+d, Σ+d], 5 .* [3 (μ - d)/μ; (μ - d)/μ d]), n ÷ 2)

# ╔═╡ 4048a66b-a89f-4e37-a89f-6fe57519d5d7
points1ₜₑₛₜ = rand(MvNormal([Σ, μ], 5 .* [2 (μ - d)/Σ; (μ - d)/Σ d]), n ÷ 2)

# ╔═╡ 17663f65-1aa1-44c4-8eae-f4bc6e24fe98
points2ₜₑₛₜ = rand(MvNormal([-μ+d, Σ+d], 5 .* [3 (μ - d)/μ; (μ - d)/μ d]), n ÷ 2)

# ╔═╡ 16390a59-9ef0-4b05-8412-7eef4dfb13ee
md"""
!!! todo
 Your task here is implement the PLA (1 point). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 43dee3c9-88f7-4c79-b4a3-6ab2cc3bba2e
"""
	Perceptron learning algorithm (PLA) implement function.

### Fields
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
- n_epochs::Int64=10000: Maximum training epochs. Default is 10000
- η::Float64=0.03: Learning rate. Default is 0.03
"""
function pla(pos_data::Matrix{Float64}, neg_data::Matrix{Float64}, 
	n_epochs::Int64=10000, η::Float64=0.03)
	# START YOUR CODE
	# Initialize weights and bias
    θ = zeros(size(pos_data, 1) + 1)  # +1 for the bias term
    X = [pos_data neg_data]
    y = [ones(size(pos_data, 2)); -ones(size(neg_data, 2))]

    # Append a row of 1s for the bias term
    X = vcat(X, ones(1, size(X, 2)))

    # Training loop
    for epoch in 1:n_epochs
        for i in 1:size(X, 2)
            if sign(dot(θ, X[:, i])) != y[i]
                θ += η * y[i] * X[:, i]
            end
        end
    end
	# END YOUR CODE
	return θ
end

# ╔═╡ 2d7dde2b-59fc-47c0-a2d0-79dcd48d8041
θₘₗ = pla(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 8e06040d-512e-4ff6-a035-f121e9d73eb4
"""
	Decision boundary visualization function for PLA

### Fields
- θ: PLA paramters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function draw_pla(θ, pos_data::Matrix{Float64}, neg_data::Matrix{Float64})
	plt = scatter(pos_data[1, :], pos_data[2, :], label="y = 1")
  	scatter!(plt, neg_data[1, :], neg_data[2, :], label="y = -1")

	b = θ[3]
	θₘₗ = θ[1:2]

	decision(x) = θₘₗ' * x + b
	
	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])

	xₘᵢₙ = minimum(map((p) -> p[1][1], D))
  	yₘᵢₙ = minimum(map((p) -> p[1][2], D))
  	xₘₐₓ = maximum(map((p) -> p[1][1], D))
 	yₘₐₓ = maximum(map((p) -> p[1][2], D))
	
	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> decision([x, y]),
			levels=[0], linestyles=:solid, label="Decision boundary", colorbar_entry=false, color=:green)
end

# ╔═╡ f40fbc75-2879-4bf8-a2ba-7b9356149dcd
# Uncomment this line below when you finish your implementation
 draw_pla(θₘₗ, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ da9c313e-4623-4370-9d5f-0560d62deb51
"""
	Calculating values for True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)

### Fields
- y_test: Actual labels
- y_pred: Predicted labels
"""
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

# ╔═╡ c0d56e33-6dcf-4675-a679-a55e7baaeea1
"""
	Evaluation function for PLA to calculate accuracy

### Fields
- θ: PLA paramters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function eval_pla(θ, pos_data, neg_data)
	n = size(pos_data, 2)
	X = vcat(hcat(pos_data, neg_data), ones(n * 2)')
	
	y_test = vcat(ones(n), -ones(n))'
	y_pred = [sign(x) for x ∈ θ' * X]

	# START YOUR CODE
	# TODO: acc, p, r, f1???
	tp, fp, tn, fn = tpfptnfn_cal(y_test, y_pred, 1)
	acc = (tp + tn) / (tp + fp + tn + fn)
	p = tp / (tp + fp)
	r = tp / (tp + fn)
	f1 =  2 * (p * r) / (p + r)
	# END YOUR CODE
	
	print(" acc: $acc\n precision: $p\n recall: $r\n f1_score: $f1\n")
	return acc, p, r, f1
end

# ╔═╡ a9d60d10-2d93-4c3e-8720-5534efd646a4
# Uncomment this line below when you finish your implementation
eval_pla(θₘₗ, points1ₜₑₛₜ, points2ₜₑₛₜ)

# ╔═╡ d4709ae3-9de5-4d46-9d95-e15fcf741bc6
md"""

### Convergence Proof

Assume that all the training instances have bounded Euclidean norms), i.e $|| \mathbf{x} || \leq R$ . Assume that exists a linear classifier in class of functions with finite parameter values that correctly classifies all the training instances. For precisely, we assume that there is some $\gamma >0$ such that $y_t(\theta^{*})^\top\mathbf{x}_t \geq \gamma$ for all $t = 1...n$.

The convergence proof is based on combining two results:
- **Result 1**: we will show that the inner product $(\theta^{*})^\top\theta^{(k)}$ increases at least linearly with each update.
"""

# ╔═╡ bd418098-edfb-4989-8bd5-23bca5059c51
md"""
!!! todo
Your task here is show the proof of result 1. (0.25 point)
"""

# ╔═╡ 8e2c8a02-471e-4321-8d8b-d25b224aa0c1
md"""
**START YOUR PROOF**

Given:

- The $\theta^*$ is the true weight vector.
- The $\theta^k$ is the weight vector at iteration $k$.
- The $\eta$ is the learning rate.
- The $y_t$ is the true label of the misclassified instance $\mathbf{x}_t$.

The $\gamma$ is a positive constant such that $y_t(\theta^{*})^\top \mathbf{x}_t \geq \gamma > 0$ for all misclassified instances.
We aim to show that the increase in $(\theta^{*})^\top\theta^{(k)}$ after an update is at least linearly related to $\eta$.

Let's consider the inner product $(\theta^{*})^\top\theta^{(k)}$ before an update as $P = (\theta^{*})^\top\theta^{(k)}$.

After an update using a misclassified instance $\mathbf{x}_t$, the updated weight vector becomes

$\theta^{(k+1)} = \theta^{(k)} + \eta y_t \mathbf{x}_t$

The inner product after the update becomes 

$Q = (\theta^{*})^\top\theta^{(k+1)} = (\theta^{*})^\top(\theta^{(k)} + \eta y_t \mathbf{x}_t)$

Using the distributive property of inner product, we get:

$Q = (\theta^{*})^\top\theta^{(k)} +\eta y_t(\theta^{*})^\top \mathbf{x}_t$

Given that $y_t(\theta^{*})^\top \mathbf{x}_t \geq \gamma$, we can rewrite the equation as: 

$Q \geq P + \eta y_t \gamma$

This shows that the increase $Q - P$ is at least linearly related to $\eta$, with a lower bound of $\eta y_t \gamma$.

Thus, $(\theta^{*})^\top\theta^{(k)}$ increases at least linearly with each update in the Perceptron Learning Algorithm.

**END YOUR PROOF**
"""

# ╔═╡ fb06ed9a-2b6a-422f-b709-1c2f782da49e
md"""
- **Result 2**: The squared norm $||\theta^{(k)}||^2$ increases at most linearly in the number of updates $k$.
"""

# ╔═╡ 721bc350-c561-4985-b212-17cfd8d11f5a
md"""
!!! todo
Your task here is show the proof of result 2. (0.25 point)
"""

# ╔═╡ 6b4452a1-1cfd-43da-8177-2aee1259bf71
md"""
**START YOUR PROOF**

We aim to demonstrate that the squared norm of $\theta^{(k)}$, denoted as $||\theta^{(k)}||^2$, increases at most linearly with the number of updates $k$.

Given the update rule for the Perceptron Learning Algorithm:

$\theta^{(k+1)} \leftarrow \theta^{(k)} + y_t\mathbf{x}_t \text{ if } y_t(\theta^{(k)})^\top\mathbf{x}_t < 0$

We can express the squared norm at iteration $k+1$ as:


$||\theta^{(k+1)}||^2 = ||\theta^{(k)} + y_t\mathbf{x}_t||^2$

$= ||\theta^{(k)}||^2 + 2(\theta^{(k)})^\top y_t\mathbf{x}_t + ||y_t\mathbf{x}_t||^2$ 

$\leq ||\theta^{(k)}||^2 + 2R \cdot ||\theta^{(k)}|| + R^2 \quad \text{(where } ||y_t\mathbf{x}_t|| \leq R \text{)}$ 

$= (||\theta^{(k)}|| + R)^2$


This inequality shows that $||\theta^{(k+1)}||^2$ is bounded by a quadratic function of $||\theta^{(k)}||$, indicating that it increases at most linearly with the number of updates $k$. Therefore, $||\theta^{(k)}||^2$ increases at most linearly with $k$.

**END YOUR PROOF**

"""

# ╔═╡ e2bde012-e641-4ee6-aaf7-fee91e0626c2
md"""
We can now combine parts 1) and 2) to bound the cosine of the angle between $\theta^{(k)}$ and $\theta^{*}$. Since cosine is bounded by one, thus

$1 \geq \frac{k\gamma}{\sqrt{kR^2}\left \| \theta^{(*)}\right \|} \leftrightarrow k \leq \frac{R^2\left \| \theta^{(*)}\right \|^2}{\gamma^2}$

By combining the two we can show that the cosine of the angle between $\theta^{(k)}$ and $\theta^{*}$ has to increase by a finite increment due to each update. Since cosine is bounded by one, it follows that we can only make a finite number of updates.
"""

# ╔═╡ cd1160d3-4603-4d18-b107-e68355fc0604
md"""
### Geometric margin & SVM Motivation

There is a question? Does $\frac{\left \| \theta^{(*)}\right \|^2}{\gamma^2}$ relate to how difficult the classification problem is? Its inverse, i.e., $\frac{\gamma^2}{\left \| \theta^{(*)}\right \|^2}$ is the smallest distance in the vector space from any samples to the decision boundary specified by $\theta^{(*)$. In other words, it serves as a measure of how well the two classes of data are separated (by a linear boundary). We call this is gemetric margin, donated by $\gamma_{geom}$. As a result, the bound on the number of perceptron updates can be written more succinctly in terms of the geometric margin $\gamma_{geom}$ (You know that man, Vapnik–Chervonenkis Dimension)

![](https://lnhutnam.github.io/assets/images_posts/pla/geometric_margin.png)

$$k \leq \left(\frac{R}{\gamma_{geom}}\right)^2$$. We note some interesting thing about the result:
- Does not depend (directly) on the dimension of the data, nor
- number of training instances

Can’t we find such a large margin classifier directly? YES, in this homework, you will do it with Support Vector Machine :)
"""

# ╔═╡ eb804ff4-806b-4a11-af51-d4c3730c84b0
md"""
## Linear Support Vector Machine (Maximum 6 points)

From the problem statement section, we are given

$\{(x_i, y_i) | x_i \in \mathbb{R}^{d}, y_i \in \{-1, 1\}\}_{i=1}^{n}$

And based on previous section, we want to find the "maximum-geometric margin" that divides the space into two parts so that the distance between the hyperplane and the nearest point from either class is maximized. Any hyperplane can be written as the set of data points $\mathbf{x}$ satisfying

$\mathbf{\theta}^\top\mathbf{x} + b = 0$
"""

# ╔═╡ 4cd4dbad-7583-4dbd-806e-b6279aafc191
md"""
### Hard-margin

The goal of SVM is to choose two parallel hyperplanes that separate the two classes of data in order to maximize the distance between them. The region defined by these two hyperplanes is known as the "margin," and the maximum-margin hyperplane is the one located halfway between them. And these hyperplane can be decribed as

$$\mathbf{\theta}^\top\mathbf{x} + b = 1 \text{(anything on or above this boundary is of one class, with label 1)}$$ and

$$\mathbf{\theta}^\top\mathbf{x} + b = -1 \text{(anything on or below this boundary is of the other class, with label -1)}$$

Geometrically, the distance between these two hyperplanes is $\frac{2}{||\mathbf{\theta}||}$
"""

# ╔═╡ 91e528df-20e4-40b1-8ec0-96b05f59f556
md"""
!!! todo
Your task here is show that the distance between these two hyperplanes is $\frac{2}{||\mathbf{\theta}||}$ (1 point). You can modify your own code in the area bounded by START YOUR PROOF and END YOUR PROOF.
"""

# ╔═╡ e8105abb-6d8b-45ee-aebf-9ccc66b72b23
md"""
**START YOUR PROOF**

Given the equations of the hyperplanes:

- First one is: $\mathbf{\theta}^\top\mathbf{x} + b = 1$ (for points on or above the boundary)

- The second one is: $\mathbf{\theta}^\top\mathbf{x} + b = -1$ (for points on or below the boundary)

The distance between two parallel hyperplanes in the form $\mathbf{\theta}^\top\mathbf{x} + b = c_1$ and $\mathbf{\theta}^\top\mathbf{x} + b = c_2$ can be calculated as the absolute difference between their constant terms divided by the Euclidean norm of the normal vector $\mathbf{\theta}$.

Lets consider $c_1 = 1$ and $c_2 = -1$. Their difference:

$Difference = |c_1 - c_2| = |1 -(-1)| = 2$

The norm of the vector $\mathbf{\theta}$ is given by $||\mathbf{\theta}||$.

Therefore, the distance between the two hyperplanes is:

Distance = $\dfrac{Difference}{||\mathbf{\theta}||} = \dfrac{2}{||\mathbf{\theta}||}$

This demonstrates that the distance between the hyperplanes is indeed $\dfrac{2}{||\mathbf{\theta}||}$


**END YOUR PROOF**
"""

# ╔═╡ aaa8faa8-be04-4886-b336-3b0482a56480
md"""
So we want to maximize the distance betweeen these two hyperplanes? Right? Equivalently, we minimize $||\mathbf{\theta}||$. We also have to prevent data points from falling into the margin, we add the following constraint: for each $i$ either

$$\mathbf{\theta}^\top\mathbf{x}_i + b \geq 1 \text{ if } y_i = 1$$ and

$$\mathbf{\theta}^\top\mathbf{x} + b \leq -1 \text{ if } y_i = -1$$

And, we can rewrite this as

$$y_i(\mathbf{\theta}^\top\mathbf{x}_i + b) \geq 1, \forall i \in \{1...n\}$$

**Finally, the optimization problem is**

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 \\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) -1 \geq 0, \forall i = 1...n \\
  \end{gather*}$$

The parameters $\theta$ and $b$ that solve this problem determine the classifier

$$\mathbf{x} \rightarrow \text{sign}(\mathbf{\theta}^\top\mathbf{x}_i + b)$$
"""

# ╔═╡ 9ca8ef1c-cb48-474a-846f-cea211437a6e
md"""
!!! todo
 Your task here is implement the hard-margin SVM solving the primal formulation using gradient descent (3 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 8522e951-c8eb-41b9-9e27-38746934547f
"""
	SVM solving the primal formulation using gradient descent (hard-margin)
### Fields
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
- η::Float64=0.03: Learning rate. Default is 0.03
- n_epochs::Int64=10000: Maximum training epochs. Default is 10000
"""
function hardmargin_svm(pos_data, neg_data, η=0.04, n_epochs=10000)
	# START YOUR CODE
	 # Combine positive and negative data
    X = [pos_data neg_data]

	y = [ones(size(pos_data, 2)); -ones(size(neg_data, 2))]
	## Create variables for the separating hyperplane w'*x = b.
	w = zeros(size(X, 1))  # Weight vector
    b = 0.0  # Bias term
	
	## Loss function

	# Train using gradient descent
	
	## For each epoch 
	for epoch in 1:n_epochs
		# Initialize error flag
        error = false
	### For each training instance ∈ D
        # Iterate over each training instance
        for i in 1:size(X, 2)
            xi = X[:, i]
            yi = y[i]

			## Update weight
			 if yi * (dot(w, xi) + b) <= 0
                w += η * yi * xi  
                b += η * yi  
                error = true  
            end
        end

		# Check for convergence
        if !error
            break
        end
    end
	# END YOUR CODE
	## Return hyperplane parameters
	return w, b  
end

# ╔═╡ d9429c3a-04aa-48a7-bd48-07ef9289e907
# Uncomment this line below when you finish your implementation
w, b = hardmargin_svm(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 0eacbb90-e3f2-46e6-a248-5657fbaeaaf3
"""
	Visualization function for SVM solving the primal formulation using gradient descent (hard-margin)

### Fields
- w & b: SVM parameters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function draw(w, b, pos_data, neg_data)
  	plt = scatter(pos_data[1, :], pos_data[2, :], label="y = 1")
  	scatter!(plt, neg_data[1, :], neg_data[2, :], label="y = -1")

	hyperplane(x)= w' * x + b

	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])

	xₘᵢₙ = minimum(map((p) -> p[1][1], D))
  	yₘᵢₙ = minimum(map((p) -> p[1][2], D))
  	xₘₐₓ = maximum(map((p) -> p[1][1], D))
 	yₘₐₓ = maximum(map((p) -> p[1][2], D))

  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[-1],
			linestyles=:dash,
			colorbar_entry=false, color=:red, label = "Negative points")
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue, label = "Positive points")
end

# ╔═╡ ed1ae566-46bd-4006-a797-106b2f176623
# Uncomment this line below when you finish your implementation
 draw(w, b, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 8ea91cb7-e2b2-4b7a-b6b2-7921c489fb98
"""
	Evaluation function for hard-margin & soft-margin SVM to calculate accuracy

### Fields
- θ: PLA paramters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function eval_svm(w, b, pos_data, neg_data)
	n = size(pos_data, 2)
	X = hcat(pos_data, neg_data)

	# Actual labels, and predicted labels
	y_test = vcat(ones(n), -ones(n))'
	y_pred = [sign(x) for x ∈ w' * X .+ b]

	# START YOUR CODE
	# TODO: acc, p, r, f1???
	tp_, fp_, tn_, fn_ = tpfptnfn_cal(y_test, y_pred, 1)
	acc = (tp_ + tn_) / (tp_ + fp_ + tn_ + fn_)
	p = tp_ / (tp_ + fp_)
	r = tp_ / (tp_ + fn_)
	f1 =  2 * (p * r) / (p + r)
	# END YOUR CODE
	
	print(" acc: $acc\n precision: $p\n recall: $r\n f1_score: $f1\n")
	
	return acc, p, r, f1
end

# ╔═╡ 5c210f2b-910f-46c9-a30e-86d20b744adb
# Uncomment this line below when you finish your implementation
eval_svm(w, b, points1ₜₑₛₜ, points2ₜₑₛₜ)

# ╔═╡ f27aadb8-b2cf-45b9-bf99-c2382d4b2213
md"""
### Soft-margin

The limitation of Hard Margin SVM is that it only works for data that can be separated linearly. In reality, however, this would not be the case. In practice, the data will almost certainly contain noise and may not be linearly separable. In this section, we will talk about soft-margin SVM (an relaxation of the optimization problem).

Basically, the trick here is very simple, we add slack variables ςᵢ to the constraint of the optimization problem.

$$y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i, \forall i = 1...n$$

The regularized optimization problem become as

$$\begin{gather*}
    \underset{\theta, b, \varsigma}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 + \sum_{i=1}^n\varsigma_i\\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i, \forall i = 1...n \\
  \end{gather*}$$

Furthermore, we ad a regularization parameter $C$ to determine how important $\varsigma$ should be. And, we got it :)

$$\begin{gather*}
    \underset{\theta, b, \varsigma}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 + C\sum_{i=1}^n\varsigma_i\\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i,\varsigma_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$
"""

# ╔═╡ 3fdaee93-9c4f-441a-9b4a-4c037f101955
md"""
!!! todo
 Your task here is implement the soft-margin SVM solving the primal formulation using gradient descent (3 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 665885b7-9dd7-4ef9-8b5b-948295c20851
"""
	SVM solving the primal formulation using gradient descent (soft-margin)
### Fields
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
- C: relaxation variable control slack variables ς
- η::Float64=0.03: Learning rate. Default is 0.03
- n_epochs::Int64=10000: Maximum training epochs. Default is 10000
"""
function softmargin_svm(pos_data, neg_data, n_epochs=10000, C=0.12, η=0.01)
	# START YOUR CODE
	# Combine positive and negative data
    X = [pos_data neg_data]
    
    # Create labels (+1 for positive, -1 for negative)
    y = [ones(size(pos_data, 2)); -ones(size(neg_data, 2))]

	## Create variables for the separating hyperplane w'*x = b.
	w = zeros(size(X, 1))  # Weight vector
    b = 0.0  # Bias term
	
	## Loss function

	# Train using gradient descent
	## For each epoch 
	for epoch in 1:n_epochs
	### For each training instance ∈ D
	# Iterate over each training instance
        for i in 1:size(X, 2)
            xi = X[:, i]
            yi = y[i]
            
            # Calculate margin (yi * (w' * xi + b))
            margin = yi * (dot(w, xi) + b)
            
	#### Calculate slack variables ς
			slack = max(0, 1 - margin)
            
	## Update weight
			w -= η * (C * w - yi * xi * (slack > 0))
			b -= η * (C * b - yi * (slack > 0))
        end
	end
	# END YOUR CODE
	## Return hyperplane parameters
	return w, b 
end

# ╔═╡ eb0f6469-a0dd-4a6b-a3c2-6916c58072a9
# Uncomment this line below when you finish your implementation
sw, sb = softmargin_svm(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ d531768a-0aef-43ae-867b-f1670211e06f
# Uncomment this line below when you finish your implementation
 draw(sw, sb, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ f79e78e5-27d6-43be-bb32-4066dba0d373
# Uncomment this line below when you finish your implementation
 eval_svm(sw, sb, points1ₜₑₛₜ, points2ₜₑₛₜ)

# ╔═╡ 547bd5c6-a9a8-472e-87fd-e83ac5aaa0d2
md"""
## Computing the SVM classifier (To get beyond 8.5 points)

We should know about some popular kernel types we could use to classify the data such as linear kernel, polynomial kernel, Gaussian, sigmoid and RBF (radial basis function) kernel.
- Linear Kernel: $K(x_i, x_j) = x_i^\top x_j$
- Polynomial kernel: $K(x_i, x_j) = (1 + x_i^\top x_j)^p$
- Gaussian: $K(x_i, x_j) = \text{exp}\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$
- Sigmoid: $K(x_i, x_j) = \text{tanh}(\beta_0x_i^\top x_j + \beta_1)^p$
- RBF kernel: $K(x_i, x_j) = \text{exp}(-\gamma||x_i - x_j||^2)$
"""

# ╔═╡ 4f882e89-589a-4eb4-a908-e5cb2ef8c829
"""
	Function for creating two spirals dataset.

	You can check the MATLAB implement here: 6 functions for generating artificial datasets, https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets
### FIELDS
- nₛₐₘₚₗₑₛ: number of samples you want :)
- noise: noise rate for creating process you want :)
"""
function two_spirals(nₛₐₘₚₗₑₛ, noise::Float64=0.2)
  start_angle = π / 2
  total_angle = 3π

  N₁ = floor(Int, nₛₐₘₚₗₑₛ / 2)
  N₂ = nₛₐₘₚₗₑₛ - N₁

  n = start_angle .+ sqrt.(rand(N₁, 1)) .* total_angle
  d₁ = [-cos.(n) .* n + rand(N₁, 1) .* noise sin.(n) .* n + rand(N₁, 1) .* noise]

  n = start_angle .+ sqrt.(rand(N₂, 1)) .* total_angle
  d₂ = [cos.(n) .* n + rand(N₂, 1) * noise -sin.(n) .* n + rand(N₂, 1) .* noise]

  return d₁', d₂'
end

# ╔═╡ 5784e0c3-4baa-4a55-8e00-6fb501fedee8
# create two spirals which are not linearly seperable
sp_points1, sp_points2 = two_spirals(500)

# ╔═╡ 6e77fe50-767b-48e3-827e-2ed9c7b91b9c
scatter!(scatter(sp_points1[1, :], sp_points1[2, :], label="y = 1"), sp_points2[1, :], sp_points2[2, :], label="y = -1")

# ╔═╡ a7d3fe4a-0367-4ef0-9816-801350fc8534
# Kernel function: in this lab, we use RBF kernel function, you want to do more experiment, please try again at home
γ = 1 / 5

# ╔═╡ 1bc5da97-cb97-4c64-9a32-f9697d6e11fe
K(x, y) = exp(-γ * (x - y)' * (x - y))

# ╔═╡ dc0d267f-4a1e-49e9-8e44-d5674771f193
md"""
### SMO algorithm 

For more detail, you should read: Platt, J. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines.

Wikipedia just quite good for describes this algorithm: MO is an iterative algorithm for solving the optimization problem. MO breaks this problem into a series of smallest possible sub-problems, which are then solved analytically. Because of the linear equality constraint involving the Lagrange multipliers $\lambda_i$, the smallest possible problem involves two such multipliers.

The SMO algorithm proceeds as follows:
- Step 1: Find a Lagrange multiplier $\alpha_1$ that violates the Karush–Kuhn–Tucker (KKT) conditions for the optimization problem.
- Step 2: Pick a second multiplier $\alpha_2$ and optimize the pair ($\alpha_1, \alpha_2$)
- Step 3: Repeat steps 1 and 2 until convergence.
"""

# ╔═╡ 18f39850-c867-4866-9389-13658f71b200
md"""
### Dual SVM - Hard-margin

If you want to find minimum of a function $f$ under the equality constraint $g$, we can use Largrangian function

$$f(x)-\lambda g(x)=0$$
where $\lambda$ is Lagrange multiplier.

In terms of SVM optimization problem

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 \\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) -1 \geq 0, \forall i = 1...n \\
  \end{gather*}$$

The equality constraint is $$g(\theta, b) = y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) -1,\forall i = 1...n$$

Then the Lagrangian function is

$$\mathcal{L}(\theta, b, \lambda) = \frac{1}{2}\left\| \theta\right\|^2 + \sum_1^n\lambda_i\left(y_i(\mathbf{\theta}^\top \mathbf{x}_i+b)-1)\right)$$

Equivalently, Lagrangian primal problem is formulated as

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }} {\text{ max }} \mathcal{L}(\theta, b, \lambda)\\
    \text{s.t.}\quad \lambda_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$

!!! note
	We need to MINIMIZE the MAXIMIZATION of $\mathcal{L}(\theta, b, \lambda)$? What we are doing???

!!! danger
	More precisely, $\lambda$ here should be KKT (Karush-Kuhn-Tucker) multipliers

	$$\lambda [-y_i\left(\theta^\top\mathbf{x}_i + b\right) + 1] = 0, \forall i = 1...n$$
"""

# ╔═╡ 730ee186-b178-401c-b274-c72493928e80
md"""
With the Lagrangian function

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }} {\text{ max }} \mathcal{L}(\theta, b, \lambda)= \frac{1}{2}\left\| \theta\right\|^2 + \sum_{i=1}^n\lambda_i\left(y_i(\mathbf{\theta}^\top \mathbf{x}_i+b-1)\right)\\
    \text{s.t.}\quad \lambda_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$

Setting derivatives to 0 yield:

$$\begin{align}
\nabla_{\mathbf{\theta}}\mathcal{L}(\theta, b, \lambda) &= \theta - \sum_{i=1}^n\lambda_iy_i\mathbf{x}_i = 0 \Leftrightarrow \mathbf{\theta}^{*} = \sum_{i=1}^n\lambda_iy_i\mathbf{x}_i \\
\nabla_b \mathcal{L}(\theta, b, \lambda) &= -\sum_{i=1}^n\lambda_iy_i = 0
\end{align}$$

We substitute them into the Lagrangian function, and get

$$W(\lambda, b) = \sum_{i=1}^n\lambda_i -\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\lambda_i\lambda_jy_iy_j\mathbf{x}_i\mathbf{x}_j$$

So, dual problem is stated as

$$\begin{gather*}
    \underset{\lambda}{\text{ max }}\sum_1^n\lambda_i -\frac{1}{2}\sum_i^n\sum_j^n\lambda_i\lambda_jy_iy_j\mathbf{x}_i\mathbf{x}_j\\
    \text{s.t.}\quad \lambda_i \geq 0, \forall i = 1...n, \sum_{i=1}^n\lambda_iyi=0 \\
  \end{gather*}$$

To solve this one has to use quadratic optimization or **sequential minimal optimization**
"""

# ╔═╡ e4a0072e-8920-4005-ba2a-a5e12a9d5f6a
function draw_nl(λ, b, pos_data, neg_data)
  	plt = scatter(pos_data[1, :], pos_data[2, :], label="y = 1")
  	scatter!(plt, neg_data[1, :], neg_data[2, :], label="y = -1")

	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])

	X = [x for (x, y) in D]
	Y = [y for (x, y) in D]

	k(x, y) = exp(-1 / 5 * (x - y)' * (x - y))

	hyperplane(x)= (λ .* Y) ⋅ k.(X, Ref(x)) + b

	xₘᵢₙ = minimum(map((p) -> p[1][1], D))
  	yₘᵢₙ = minimum(map((p) -> p[1][2], D))
  	xₘₐₓ = maximum(map((p) -> p[1][1], D))
 	yₘₐₓ = maximum(map((p) -> p[1][2], D))

  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[-1],
			linestyles=:dash,
			colorbar_entry=false, color=:red, label = "Negative points")
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue, label = "Positive points")
end

# ╔═╡ bcc10780-3058-46fa-9123-79b0d0861e0d
md"""
!!! todo
 Your task here is implement the hard-margin SVM solving the dual formulation using sequential minimal optimization (2 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 6b7d6bf7-afcf-4dce-8488-b97509ef8e88
function dualsvm_smo_hard(pos_data, neg_data, n_epochs=100, λₜₒₗ=0.0001, errₜₒₗ=0.0001)
	# You do not need implement kernel, please use the K(.) kernel function in previous cell code.
	
	# START YOUR CODE
	# Step 1: Data preparation
	# First you construct and shuffle to obtain dataset D in a stochastically manner
	D = [(x, 1) for x in eachcol(pos_data)]  
    append!(D, [(x, -1) for x in eachcol(neg_data)])  

	D = shuffle(D)
	# For more easily access to data point
	X = [x for (x, y) ∈ D]
	Y = [y for (x, y) ∈ D]
    
	
	# Step 2: Initialization
	# Larangian multipliers, and bias
	λ = zeros(length(D))
	b = 0
	n = length(λ)
	C = Inf
	
	# Step 3: Training loop
	for epoch in 1:n_epochs
        num_changed_alphas = 0
        for i in 1:n
            E_i = b
            for j in 1:n
                E_i += λ[j] * Y[j] * dot(X[i], X[j])
            end
            E_i -= Y[i]

            if (Y[i] * E_i < -errₜₒₗ && λ[i] < C) || (Y[i] * E_i > errₜₒₗ && λ[i] > 0)
                j = rand(1:n)  # Randomly select second alpha (j)
                while j == i
                    j = rand(1:n)
                end

                E_j = b
                for k in 1:n
                    E_j += λ[k] * Y[k] * dot(X[j], X[k])
                end
                E_j -= Y[j]

                old_λ_i, old_λ_j = copy(λ[i]), copy(λ[j])

                if Y[i] != Y[j]
                    L = max(0, λ[j] - λ[i])
                    H = min(C, C + λ[j] - λ[i])
                else
                    L = max(0, λ[i] + λ[j] - C)
                    H = min(C, λ[i] + λ[j])
                end

                if L == H
                    continue
                end

                eta = 2 * dot(X[i], X[j]) - dot(X[i], X[i]) - dot(X[j], X[j])
                if eta >= 0
                    continue
                end

                λ[j] -= Y[j] * (E_i - E_j) / eta
                λ[j] = max(L, min(λ[j], H))

                if abs(λ[j] - old_λ_j) < errₜₒₗ
                    continue
                end

                λ[i] += Y[i] * Y[j] * (old_λ_j - λ[j])
                b1 = b - E_i - Y[i] * (λ[i] - old_λ_i) * dot(X[i], X[i]) - Y[j] * (λ[j] - old_λ_j) * dot(X[i], X[j])
                b2 = b - E_j - Y[i] * (λ[i] - old_λ_i) * dot(X[i], X[j]) - Y[j] * (λ[j] - old_λ_j) * dot(X[j], X[j])
                b = (b1 + b2) / 2

                num_changed_alphas += 1
            end
        end

        if num_changed_alphas == 0
            break
        end
    end
	# END YOUR CODE
	## Return hyperplane parameters
	return λ, b
end

# ╔═╡ c5028050-48ac-4e07-9a6c-e836537ff7c7
# Uncomment this line below when you finish your implementation
λₕ, bₕ = dualsvm_smo_hard(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 52128a2f-5a4f-4e11-ad2b-e112098b8b82
# Uncomment this line below when you finish your implementation
draw_nl(λₕ, bₕ, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ d14d2d72-8c39-462d-b30f-8e1e4765159e
md"""
### Dual SVM - Soft-margin

As we know that, the regularized optimization problem in the case of soft-margin as

$$\begin{gather*}
    \underset{\theta, b, \varsigma}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 + C\sum_{i=1}^n\varsigma_i\\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i,\varsigma_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$

We use Larangian multipliers, and transform to a dual problem as 

$$\begin{gather*}
    \underset{\lambda}{\text{ max }}\sum_1^n\lambda_i -\frac{1}{2}\sum_i^n\sum_j^n\lambda_i\lambda_jy_iy_j\mathbf{x}_i\mathbf{x}_j\\
    \text{s.t.}\quad  0 \leq \lambda_i \leq C, \forall i = 1...n, \sum_{i=1}^n\lambda_iyi=0 \\
  \end{gather*}$$
"""

# ╔═╡ fbc7b96a-67ae-46b3-b746-4ea50a4455ce
md"""
!!! todo
 Your task here is implement the soft-margin SVM solving the dual formulation using sequential minimal optimization (2 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ e75a6b8a-9e34-4b1b-9bd2-7641454f12c0
function dualsvm_smo_soft(pos_data, neg_data, n_epochs=100, C=1000, λₜₒₗ=0.0001, errₜₒₗ=0.0001)
	# START YOUR CODE
	
	# Step 1: Data preparation
	# First you construct and shuffle to obtain dataset D in a stochastically manner
	D = [(x, 1) for x in eachcol(pos_data)]
    append!(D, [(x, -1) for x in eachcol(neg_data)])
	
    D = shuffle(D)
	# For more easily access to data point
	X = [x for (x, y) ∈ D]
	Y = [y for (x, y) ∈ D]

	# Step 2: Initialization
	# Larangian multipliers, and bias
	λ = zeros(length(D))
	b = 0
	n = length(λ)

	# Step 3: Training loop
	for epoch in 1:n_epochs
        num_changed_alphas = 0
        for i in 1:n
            E_i = b
            for j in 1:n
                E_i += λ[j] * Y[j] * dot(X[i], X[j])
            end
            E_i -= Y[i]

            if (Y[i] * E_i < -errₜₒₗ && λ[i] < C) || (Y[i] * E_i > errₜₒₗ && λ[i] > 0)
                j = rand(1:n)  # Randomly select second alpha (j)
                while j == i
                    j = rand(1:n)
                end

                E_j = b
                for k in 1:n
                    E_j += λ[k] * Y[k] * dot(X[j], X[k])
                end
                E_j -= Y[j]

                old_λ_i, old_λ_j = copy(λ[i]), copy(λ[j])

                if Y[i] != Y[j]
                    L = max(0, λ[j] - λ[i])
                    H = min(C, C + λ[j] - λ[i])
                else
                    L = max(0, λ[i] + λ[j] - C)
                    H = min(C, λ[i] + λ[j])
                end

                if L == H
                    continue
                end

                eta = 2 * dot(X[i], X[j]) - dot(X[i], X[i]) - dot(X[j], X[j])
                if eta >= 0
                    continue
                end

                λ[j] -= Y[j] * (E_i - E_j) / eta
                λ[j] = max(L, min(λ[j], H))

                if abs(λ[j] - old_λ_j) < errₜₒₗ
                    continue
                end

                λ[i] += Y[i] * Y[j] * (old_λ_j - λ[j])
                b1 = b - E_i - Y[i] * (λ[i] - old_λ_i) * dot(X[i], X[i]) - Y[j] * (λ[j] - old_λ_j) * dot(X[i], X[j])
                b2 = b - E_j - Y[i] * (λ[i] - old_λ_i) * dot(X[i], X[j]) - Y[j] * (λ[j] - old_λ_j) * dot(X[j], X[j])
                b = (b1 + b2) / 2

                num_changed_alphas += 1
            end
        end

        if num_changed_alphas == 0
            break
        end
    end
	# END YOUR CODE
	## Return hyperplane parameters
	return λ, b
end

# ╔═╡ 2d29d23f-7463-4d88-8318-fdcb78bacd3f
# Uncomment this line below when you finish your implementation
λₛ, bₛ = dualsvm_smo_soft(sp_points1, sp_points2)

# ╔═╡ 438aea80-21a7-4e56-aaa3-6f8b4dabc976
# Uncomment this line below when you finish your implementation
draw_nl(λₛ, bₛ, sp_points1, sp_points2)

# ╔═╡ 25054281-405d-458f-ab3a-e05f1f956bec
md"""
## Multi-classes classification problem with SVMs (To get beyond 10.0 points)
"""

# ╔═╡ ed31489c-3feb-483d-9787-87df73e116d0
md"""
### Load MNIST dataset
"""

# ╔═╡ 513a10db-cc97-4a6c-b7b3-eee6b6c283f4
begin
	data_dir = joinpath(dirname(@__FILE__), "data")
	train_x_dir = joinpath(data_dir, "train/images/train-images.idx3-ubyte")
	train_y_dir = joinpath(data_dir, "train/labels/train-labels.idx1-ubyte")
	
	test_x_dir = joinpath(data_dir, "test/images/t10k-images.idx3-ubyte")
	test_y_dir = joinpath(data_dir, "test/labels/t10k-labels.idx1-ubyte")
	
	NUMBER_TRAIN_SAMPLES = 60000
	NUMBER_TEST_SAMPLES = 10000
end

# ╔═╡ 8d004f4b-5523-4414-9ca9-a5509d541236
begin
	train_x = Array{Float64}(undef, 28^2, NUMBER_TRAIN_SAMPLES)
	train_y = Array{Int64}(undef, NUMBER_TRAIN_SAMPLES)

	io_images = open(train_x_dir)
	io_labels = open(train_y_dir)

	for i ∈ 1:NUMBER_TRAIN_SAMPLES
		seek(io_images, (i-1)*28^2 + 16) # offset 16 to skip header
		seek(io_labels, (i-1)*1 + 8) # offset 8 to skip header
		train_x[:,i] = convert(Array{Float64}, read(io_images, 28^2))
		train_y[i] = convert(Int, read(io_labels, UInt8))
	end
	close(io_images)
	close(io_labels)

	train_x = train_x
end

# ╔═╡ bead671f-1f61-44ed-ba4c-0b4156757faa
begin
	test_x = Array{Float64}(undef, 28^2, NUMBER_TEST_SAMPLES)
	test_y = Array{Int64}(undef, NUMBER_TEST_SAMPLES)

	io_images_test = open(test_x_dir)
	io_labels_test = open(test_y_dir)

	for i ∈ 1:NUMBER_TEST_SAMPLES
		seek(io_images_test, (i-1)*28^2 + 16) # offset 16 to skip header
		seek(io_labels_test, (i-1)*1 + 8) # offset 8 to skip header
		test_x[:,i] = convert(Array{Float64}, read(io_images_test, 28^2))
		test_y[i] = convert(Int, read(io_labels_test, UInt8))
	end
	close(io_images)
	close(io_labels)

	test_x = test_x
end

# ╔═╡ b949cfb8-c649-46d5-8d9a-47a0a153fe3a
size(train_x), size(train_y), size(test_x), size(test_y)

# ╔═╡ d52bb268-787a-4590-ba7a-699e23a93092
md"""
### Training SVMs
"""

# ╔═╡ ddef06cb-469e-45cd-bd6c-d5796b1da64d
# STAR YOUR CODE
begin
	# Step 1: Preprocessing
	function preprocessing(train_x, test_x)
		# Normalize data
		train_x ./= 255.0
		test_x ./= 255.0
		
		# Reshape data for processing
		train_x = reshape(train_x, (28*28, :))
		test_x = reshape(test_x, (28*28, :))
		
		return train_x, test_x
	end
	
	# Step 2: Hinge Loss Calculation
	function hinge_loss(X, y, θ, b)
	    # Hinge loss
	    loss = 0.0
	    for i in 1:size(X, 2)
	        loss += max(0, 1 - y[i] * (dot(θ, X[:, i]) + b))
	    end
	    return loss / size(X, 2)
	end
	
	# Step 3: Gradient Descent
	function gradient_descent(X, y, θ, b, learning_rate)
	    # Gradient calculation
	    dθ = zeros(size(X, 1))
	    db = 0.0
	    for i in 1:size(X, 2)
	        if y[i] * (dot(θ, X[:, i]) + b) < 1
	            dθ -= y[i] * X[:, i]
	            db -= y[i]
	        end
	    end
	    
	    # Update parameters
	    θ -= learning_rate * (dθ / size(X, 2))
	    b -= learning_rate * (db / size(X, 2))
	    
	    return θ, b
	end
	
	# Step 4: Training Loop
	function training_svms(train_x, train_y, epochs, learning_rate)
		θ = zeros(size(train_x, 1))  # Weight vector
		b = 0                        # Bias term
		losses = []
		
		for epoch in 1:epochs
		    θ, b = gradient_descent(train_x, train_y, θ, b, learning_rate)
		    loss = hinge_loss(train_x, train_y, θ, b)
			append!(losses, loss)
		    println("Epoch: $epoch, Loss: $loss")
		end
		
		return θ, b, losses
	end
end
# END YOUR CODE

# ╔═╡ 96a52556-f272-4f31-a636-1d0be26fc43d
train_x_pre, test_x_pre = preprocessing(train_x, test_x)

# ╔═╡ b252885b-46b1-4d83-9421-bd4c6b875c3c
begin
	learning_rate = 0.01
	epochs = 100
	θ_train, b_train, losses = training_svms(train_x_pre, train_y, epochs, learning_rate)
end

# ╔═╡ fd7711dc-4b75-4462-b553-e6d843993202
plot(1:epochs, losses, legend=false)

# ╔═╡ 6cc636b3-8dfa-4069-89e5-81cc0a500e8e
md"""
### Evaluation
"""

# ╔═╡ 3046e2e5-7df5-40c8-b621-e3907c4477a2
# START YOUR CODE
function accuracy(X, y, θ, b)
    correct = 0
    for i in 1:size(X, 2)
        if sign(dot(θ, X[:, i]) + b) == y[i]
            correct += 1
        end
    end
    return correct / size(X, 2)
end
# END YOUR CODE

# ╔═╡ afbe003c-1ff8-4554-b973-3216146c03b7
# Evaluate on the training set
begin
	train_accuracy = accuracy(train_x, train_y, θ_train, b_train)
	println("Training Accuracy: $(train_accuracy * 100)%")
	
	# Evaluate on the test set
	test_accuracy = accuracy(test_x, test_y, θ_train, b_train)
	println("Test Accuracy: $(test_accuracy * 100)%")
end

# ╔═╡ 6771c4f1-cf02-4a72-8ffc-b78b00514428
md"""
This is the end of Lab 05. However, there still a lot of things that you can learn about SVM. There are many open tasks to do in your sparse time such as how to deal with multi-class, or Bayesian SVM. :) Hope all you will enjoy SVM. Good luck!
"""

# ╔═╡ 488098f8-1881-459f-aaef-df1a59058b73
md"""
## References

[1] Boyd, S. P., & Vandenberghe, L. (2004). Convex optimization. Cambridge university press.

[2] Griva, I., Nash, S. G., & Sofer, A. (2008). Linear and Nonlinear Optimization 2nd Edition. Society for Industrial and Applied Mathematics.

[3] Schölkopf, B., & Smola, A. J. (2002). Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press.

[4] Lab 3, Logistic Regresion, Introduction to Machine Learning course, Department of Computer Science, Faculty of Information Technology, Ho Chi Minh University of Science, Vietnam National University.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.80"
Plots = "~1.38.5"
PlutoUI = "~0.7.50"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

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
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "74911ad88921455c6afcad1eefa12bd7b1724631"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.80"

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

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

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
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

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
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

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
git-tree-sha1 = "660b2ea2ec2b010bb02823c6d0ff6afd9bdc5c16"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.7"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d5e1fd17ac7f3aa4c5287a61ee28d4f8b8e98873"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.7+0"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

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
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

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
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

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
git-tree-sha1 = "071602a0be5af779066df0d7ef4e14945a010818"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.22"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

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
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

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
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "8ac949bd0ebc46a44afb1fdca1094554a84b086e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.5"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

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
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

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
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

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

[[SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

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
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

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
git-tree-sha1 = "f7c281e9c61905521993a987d38b5ab1d4b53bef"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+1"

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
# ╟─287aba39-66d9-4ff6-9605-1bca094a1ce5
# ╟─eeae9e2c-aaf8-11ed-32f5-934bc393b3e6
# ╟─86f24f32-d8ee-49f2-b71a-bd1c5cd7a28f
# ╟─cab7c88c-fe3f-40c4-beea-eea924d67975
# ╠═08c5b8cf-be45-4d11-bd24-204b364a8278
# ╠═99329d11-e709-48f0-96b5-32ae0cac1f50
# ╟─bbccfa2d-f5b6-49c7-b11e-53e419808c1b
# ╟─4321f49b-1057-46bc-8d67-be3122be7a68
# ╟─4fdaeeda-beee-41e1-a5f0-3209151a880d
# ╟─ec906d94-8aed-4df1-932c-fa2263e6325d
# ╟─42882ca3-8df6-4fb8-8884-29337373bac5
# ╠═e78094ff-6565-4e9d-812e-3a36f78731ed
# ╠═d0ecb21c-6189-4b10-9162-1b94424f49ce
# ╠═921e8d15-e751-4976-bb80-2cc09e6c950e
# ╠═4048a66b-a89f-4e37-a89f-6fe57519d5d7
# ╠═17663f65-1aa1-44c4-8eae-f4bc6e24fe98
# ╟─16390a59-9ef0-4b05-8412-7eef4dfb13ee
# ╠═43dee3c9-88f7-4c79-b4a3-6ab2cc3bba2e
# ╠═2d7dde2b-59fc-47c0-a2d0-79dcd48d8041
# ╠═8e06040d-512e-4ff6-a035-f121e9d73eb4
# ╠═f40fbc75-2879-4bf8-a2ba-7b9356149dcd
# ╠═da9c313e-4623-4370-9d5f-0560d62deb51
# ╠═c0d56e33-6dcf-4675-a679-a55e7baaeea1
# ╠═a9d60d10-2d93-4c3e-8720-5534efd646a4
# ╟─d4709ae3-9de5-4d46-9d95-e15fcf741bc6
# ╟─bd418098-edfb-4989-8bd5-23bca5059c51
# ╟─8e2c8a02-471e-4321-8d8b-d25b224aa0c1
# ╟─fb06ed9a-2b6a-422f-b709-1c2f782da49e
# ╠═721bc350-c561-4985-b212-17cfd8d11f5a
# ╟─6b4452a1-1cfd-43da-8177-2aee1259bf71
# ╟─e2bde012-e641-4ee6-aaf7-fee91e0626c2
# ╟─cd1160d3-4603-4d18-b107-e68355fc0604
# ╟─eb804ff4-806b-4a11-af51-d4c3730c84b0
# ╟─4cd4dbad-7583-4dbd-806e-b6279aafc191
# ╟─91e528df-20e4-40b1-8ec0-96b05f59f556
# ╟─e8105abb-6d8b-45ee-aebf-9ccc66b72b23
# ╟─aaa8faa8-be04-4886-b336-3b0482a56480
# ╟─9ca8ef1c-cb48-474a-846f-cea211437a6e
# ╠═8522e951-c8eb-41b9-9e27-38746934547f
# ╠═d9429c3a-04aa-48a7-bd48-07ef9289e907
# ╠═0eacbb90-e3f2-46e6-a248-5657fbaeaaf3
# ╠═ed1ae566-46bd-4006-a797-106b2f176623
# ╠═8ea91cb7-e2b2-4b7a-b6b2-7921c489fb98
# ╠═5c210f2b-910f-46c9-a30e-86d20b744adb
# ╟─f27aadb8-b2cf-45b9-bf99-c2382d4b2213
# ╟─3fdaee93-9c4f-441a-9b4a-4c037f101955
# ╠═665885b7-9dd7-4ef9-8b5b-948295c20851
# ╠═eb0f6469-a0dd-4a6b-a3c2-6916c58072a9
# ╠═d531768a-0aef-43ae-867b-f1670211e06f
# ╠═f79e78e5-27d6-43be-bb32-4066dba0d373
# ╠═547bd5c6-a9a8-472e-87fd-e83ac5aaa0d2
# ╠═4f882e89-589a-4eb4-a908-e5cb2ef8c829
# ╠═5784e0c3-4baa-4a55-8e00-6fb501fedee8
# ╠═6e77fe50-767b-48e3-827e-2ed9c7b91b9c
# ╠═a7d3fe4a-0367-4ef0-9816-801350fc8534
# ╠═1bc5da97-cb97-4c64-9a32-f9697d6e11fe
# ╟─dc0d267f-4a1e-49e9-8e44-d5674771f193
# ╠═18f39850-c867-4866-9389-13658f71b200
# ╟─730ee186-b178-401c-b274-c72493928e80
# ╠═e4a0072e-8920-4005-ba2a-a5e12a9d5f6a
# ╟─bcc10780-3058-46fa-9123-79b0d0861e0d
# ╠═6b7d6bf7-afcf-4dce-8488-b97509ef8e88
# ╠═c5028050-48ac-4e07-9a6c-e836537ff7c7
# ╠═52128a2f-5a4f-4e11-ad2b-e112098b8b82
# ╠═d14d2d72-8c39-462d-b30f-8e1e4765159e
# ╟─fbc7b96a-67ae-46b3-b746-4ea50a4455ce
# ╠═e75a6b8a-9e34-4b1b-9bd2-7641454f12c0
# ╠═2d29d23f-7463-4d88-8318-fdcb78bacd3f
# ╠═438aea80-21a7-4e56-aaa3-6f8b4dabc976
# ╠═25054281-405d-458f-ab3a-e05f1f956bec
# ╠═ed31489c-3feb-483d-9787-87df73e116d0
# ╠═513a10db-cc97-4a6c-b7b3-eee6b6c283f4
# ╠═8d004f4b-5523-4414-9ca9-a5509d541236
# ╠═bead671f-1f61-44ed-ba4c-0b4156757faa
# ╠═b949cfb8-c649-46d5-8d9a-47a0a153fe3a
# ╠═d52bb268-787a-4590-ba7a-699e23a93092
# ╠═ddef06cb-469e-45cd-bd6c-d5796b1da64d
# ╠═96a52556-f272-4f31-a636-1d0be26fc43d
# ╠═b252885b-46b1-4d83-9421-bd4c6b875c3c
# ╠═fd7711dc-4b75-4462-b553-e6d843993202
# ╠═6cc636b3-8dfa-4069-89e5-81cc0a500e8e
# ╠═3046e2e5-7df5-40c8-b621-e3907c4477a2
# ╠═afbe003c-1ff8-4554-b973-3216146c03b7
# ╟─6771c4f1-cf02-4a72-8ffc-b78b00514428
# ╟─488098f8-1881-459f-aaef-df1a59058b73
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
