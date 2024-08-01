import pandas as pd
import unittest

class TestDataIntegrity(unittest.TestCase):

    def test_no_null_values(self):
        tweet_df = pd.read_csv('train.csv')  
        self.assertFalse(tweet_df.isnull().values.any(), "Data contains null values.")
        

if __name__ == '__main__':
    unittest.main()
    
