

	def load_connect_pandas(results):
        df = pd.DataFrame(list(results))
        df['CollectDate'] = df['MetadataLocal'].apply(lambda x: x.get('CollectDate') if isinstance(x, dict) else None))
        df['ProjectId'] = df['MetadataLocal'].apply(lambda x: x.get('ProjectId') if isinstance(x, dict) else None)
        
    def plot_scatter_plot(bydayCost, dailyCost, collectDate, projectIds):
        np.random.seed(10)
        data = pd.DataFrame({
        	'SeriesA': np.ramdom.normal(loc=0, scale=1, size=100)
            'bbb': np.random.choice(['X','Y','Z'], 100)
        })
        data['bydayCost'] = bydayCost
        plt.figure(figsize=(10,6))
        sns.scatterplot(x='projectIds', y='bydayCost', hue='bbb', data=data, palette='viridis', s=100)
        plt.title('test')
        plt.xlabel('byday')
        plt.ylabel('daily')
        plt.legend(title='category')
        plt.show()