MPTCP is a transport mechanism that allows peers to use multiple TCP subflows via their existing IP addresses at the same time. Each
subflow encounters the bottleneck connection status of its path, so scheduling an outgoing packet with the best subflow is crucial to multipath performance. Poor scheduling decisions prevent users from utilizing the pooling potential of available subflows, even though good
scheduling decisions can significantly boost throughput. DRL algorithms like Model Agnostic Meta Learning are used to analyze packet scheduling in the MPTCP in detail to address this issue. These algorithms can help optimize the decision-making process by taking into account observed network states and rewards.
Model-Agnostic Meta-Learning (MAML), is designed to create models that can quickly adapt to new tasks or environments with minimal retraining. This algorithm is particularly
powerful in scenarios where the environment is dynamic and unpredictable. MAML is a Meta-RL algorithm that can quickly the scheduler's
policy to optimize packet transfer across paths. The scheduler doesn't just learn how to route packets in one specific scenario but learns a strategy that works well across a wide range of network conditions.
This algorithm has been implemented in Python and Golang.
