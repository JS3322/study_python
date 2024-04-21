import os
import yaml
from pymongo import MongoClient

class readMongodb:

	with open('application.yml', 'r') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
		os.environ['MONGODB_HOST'] = cfg['spring']['data']['mongodb_dev']['host']
		os.environ['MONGODB_PORT'] = str(cfg['spring']['data']['mongodb_dev']['port'])

	def load_connect_mongodb():
		print(os.environ.get('MONGODB_HOST'))

		client = MongoClient('mongodb://ACCOUNT:PASSWORD'+os.environ.get('MONGODB_HOST')+':'+os.environ.get('MONGODB_PORT'))
		db = client['test']
		collection = db['TEST_COLLECTION']
		pipeline = [{'$match':{'MetadataLocal.ProductName':'EC2'}}]
		results = collection.aggregate(pipeline)
		for result in results:
			print(result)