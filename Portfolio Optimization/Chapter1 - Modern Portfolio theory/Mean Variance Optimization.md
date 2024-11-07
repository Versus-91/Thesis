[[Background]]

paper:
Portfolio Selection
link: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1952.tb01525.x
![[Pasted image 20241013133332.png]]Mean variance optimization is a classic Portfolio selection method which was introduced in 1952. 
MVO and similar methods such as Mean Absolute, Black-Litterman, and etc.  The main idea behind MVO is by combining various assets with different expected returns and volatilities it is possible to find an optimal mathematic allocation. 

MVO works solely with historical data without any need for market macro economic data. In order to find MVO we need to use expected returns and covariance matrix of the assets. While there is no optimal way for estimating expected return it is possible to use mean of historical returns or other proprietary methods.

Using MVO would enable users to find optimal combination of assets based on the desired expected returns and risk but this method struggles with high dimensional spaces and situations where the returns are not following a normal distribution. The biggest flaw of this method is reliance on expected returns, which are difficult to estimate.


**MVO limitations

MVO performs well when asset returns are normally distributed, however in real world scenarios returns rarely follow normal distribution, making MVO less reliable. furthermore accurately estimating returns and covariances is challenging. using simple historical mean of returns is not able to capture complexity of financial markets and often leads to suboptimal portfolios. While there are various methods for estimating expected returns of assets, it is considered not using expected returns for portfolio optimization is ideal.

Paper : **Optimal versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?



