# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:43:03 2024

@author: Gabin
"""

import requests
import requests.auth

client_auth = requests.auth.HTTPBasicAuth('p-jcoLKBynTLew', 'gko_LXELoV07ZBNUXrvWZfzE3aI')
post_data = {"grant_type": "password", "username": "reddit_bot", "password": "snoo"}
headers = {"User-Agent": "ChangeMeClient/0.1 by YourUsername"}
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
response.json()

#Example response
# {u'access_token': u'fhTdafZI-0ClEzzYORfBSCR7x3M',
#  u'expires_in': 3600,
#  u'scope': u'*',
#  u'token_type': u'bearer'}

# url="https://oauth.reddit.com/api/v1/me"
url = "https://oauth.reddit.com/r/datascience/top"
headers = {"Authorization": "bearer fhTdafZI-0ClEzzYORfBSCR7x3M",
           "User-Agent": "ChangeMeClient/0.1 by YourUsername",
           "t": "hour",
           "limit": 10
           }

response = requests.get(url, headers=headers)
response.json()

# example response
# Out[10]: 
# {u'comment_karma': 0,
#  u'created': 1389649907.0,
#  u'created_utc': 1389649907.0,
#  u'has_mail': False,
#  u'has_mod_mail': False,
#  u'has_verified_email': None,
#  u'id': u'1',
#  u'is_gold': False,
#  u'is_mod': True,
#  u'link_karma': 1,
#  u'name': u'reddit_bot',
#  u'over_18': True}