#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:16:08 2019

@author: deepank
"""

import os
import time
from selenium import webdriver
driver = webdriver.Chrome()
driver.implicitly_wait(10)
driver.get('https://competition.bitgrit.net/competition/3#')
           
for r,d,f in os.walk('/home/deepank/Downloads/BITGRIT/Upload/'):
    for x in f: 
        if '.py 'not  in x:
            print(x)
            time.sleep(1)
            driver.find_element_by_id('first-submission').click()
            time.sleep(1)
            driver.find_element_by_xpath('//input[@accept=".csv"]').send_keys('/home/deepank/Downloads/BITGRIT/Upload/'+x)
            time.sleep(2)
            driver.find_element_by_xpath('//input[@type="submit"]').click()
            time.sleep(2)
            driver.find_element_by_id('btn-submission').click()
            