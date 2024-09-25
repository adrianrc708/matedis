import quantstats as qs

qs.extend_pandas()

stock=qs.utils.download_returns('TSLA')

print(stock.sharpe())
stock.plot_earnings(savefig='tsla', start_balance=1000)

# qs.reports.html(stock, 'SPY', title='TSLA', output=)

