# Input: The instance V, a monotonic function f: V -> [D], the privacy budget epsilon, and failure probability beta
# Output: A privatized f(V)

# Compute tau
tau = math.ceil(2 * epsilon * math.log((D + 1) / beta))

# Compute f_hat(V, j) for j in [2*tau]
f_hat = {}
for j in range(2 * tau):
    f_hat[j] = min([f(V_bar) for V_bar in V if (V_bar <= V) and (d(V, V_bar) <= j)])

# Compute s(V, r) for r in [D]
s = {}
for r in range(D):
    if r in range(f_hat[j], f_hat[j - 1]) for some tau < j <= 2 * tau:
        s[r] = tau - j
    elif r == f_hat[tau]:
        s[r] = 0
    elif r in range(f_hat[j], f_hat[j - 1]) for some 0 < j <= tau:
        s[r] = -tau + j - 1
    else:
        s[r] = -tau - 1

# Sample r from [D] with probability proportional to exp(epsilon / 2 * s(V, r)), denoted by r_tilde
r_tilde = np.random.choice(D, p=[np.exp(epsilon / 2 * s[V, r]) for r in range(D)])

# Return M(V) = r_tilde
return r_tilde
