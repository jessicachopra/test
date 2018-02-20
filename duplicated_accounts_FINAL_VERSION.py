import pandas as pd
import os

"""
******************************************************************************
PART 1: 
 - Import the accounts data from Salesforce Duplicated_Account Report
 - Clean records
 - Split dataset into Dealer and Others parts

"""
directory = os.path.join('//wfsmnsv2nsf24/C_WBGEQF_Groups/COMMERCIAL_OFFICE/SFDC/Duplicated Accounts/')
my_file='accounts.txt' # read csv file into python. 
df1=pd.read_table(os.path.join(directory,my_file),encoding='latin-1')
df1.head()

print(df1.columns)
df1.dtypes
#have a separate dataset to reference back
account=df1
#remove special symbols from column names and rename the columns 

df1.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
df1.rename(columns=lambda x: x.replace('/', '_'), inplace=True)

#convert Account_Name, Legal_Street, Legal_City, and Legal_State into uppercase
#remove space from CCAN_MM, CCAN_ST, WCIS_ID, and DUNS columns and convert them
#into numerical data types

df1['Account_Name']=df1['Account_Name'].str.replace(',','')
df1['Account_Name']=df1['Account_Name'].str.replace('.','')
df1['Account_Name']=df1['Account_Name'].str.replace('$','')
df1['Account_Name']=df1['Account_Name'].str.replace('*','')
df1['Account_Name']=df1['Account_Name'].str.replace('|','')
df1['Account_Name']=df1['Account_Name'].str.replace('#','')
df1['Account_Name']=df1['Account_Name'].str.replace('[','')
df1['Account_Name']=df1['Account_Name'].str.replace(']','')
df1['Account_Name']=df1['Account_Name'].str.replace('"','')
df1['Account_Name']=df1['Account_Name'].astype(str).str.upper()
df1['Legal_Street']=df1['Legal_Street'].astype(str).str.upper()
df1['Legal_City']=df1['Legal_City'].astype(str).str.upper()
df1['Legal_State_Province']=df1['Legal_State_Province'].astype(str).str.upper()
df1['Dealer_Code']=df1['Dealer_Code'].str.replace(' ','')
df1['CCAN_(MM)']=df1['CCAN_(MM)'].str.replace(' ','')
df1['CCAN_(ST)']=df1['CCAN_(ST)'].str.replace(' ','')
#df1['DUNS_Number']=df1['DUNS_Number'].str.replace(' ','')
df1['CCAN_(MM)'] = pd.to_numeric(df1['CCAN_(MM)'], errors='coerce')
df1['CCAN_(ST)'] = pd.to_numeric(df1['CCAN_(ST)'], errors='coerce')
df1.dtypes

df1=df1.sort(['Account_Name','Fed_ID___SSN','Dealer_Code','Customer_Code','CCAN_(MM)','CCAN_(ST)','DUNS_Number','WCIS_ID'], ascending=[True,False,False,False,False,False,False,False])

# Check Duplicated Customer Code, WCIS_ID, CCAN_MM, CCAN_ST, and DUNS records
df1['Matching_FedId'] = df1.duplicated(['Fed_ID___SSN'])
df1['Matching_CCode'] = df1.duplicated(['Customer_Code'])
df1['Matching_SCode'] = df1.duplicated(['Shop_Code'])
df1['Matching_DCode'] = df1.duplicated(['Dealer_Code'])
df1['Matching_WCIS'] = df1.duplicated(['WCIS_ID'])
df1['Matching_CCAN_MM'] = df1.duplicated(['CCAN_(MM)'])
df1['Matching_CCAN_ST'] = df1.duplicated(['CCAN_(ST)'])
df1['Matching_DUNS'] = df1.duplicated(['DUNS_Number'])
df1['Duplicated'] = df1.duplicated(['Account_Name','Legal_Street','Legal_City','Legal_State_Province'])

# identify null records in the key identifiers
df1['null_Fed_ID']=df1['Fed_ID___SSN'].isnull()
df1['null_Customer_Code']=df1['Customer_Code'].isnull() 
df1['null_Dealer_Code']=df1['Dealer_Code'].isnull()
df1['null_Shop_Code']=df1['Shop_Code'].isnull()
df1['null_CCAN_MM']=df1['CCAN_(MM)'].isnull() 
df1['null_CCAN_ST']=df1['CCAN_(ST)'].isnull() 
df1['null_DUNS_Number']=df1['DUNS_Number'].isnull() 
df1['null_WCIS']=df1['WCIS_ID'].isnull() 
df1['null_CCAN_MM_above']=df1['null_CCAN_MM'].shift(1) 
df1['null_CCAN_ST_above']=df1['null_CCAN_ST'].shift(1) 
df1['null_DUNS_above']=df1['null_DUNS_Number'].shift(1) 
df1['CCAN_MM_above']=df1['CCAN_(MM)'].shift(1) 
df1['CCAN_ST_above']=df1['CCAN_(ST)'].shift(1) 
df1['DUNS_Number_above']=df1['DUNS_Number'].shift(1) 



# Split the dataset into Dealer/Vendor and others. Others include Entity/Corporation
#Verified account, Individual Accounts, and 3rd party accounts

dealer_vendor=df1[df1['Account_Record_Type']=='Dealer/Vendor']
others=df1[df1['Account_Record_Type']!='Dealer/Vendor']

"""
*******************************************************************************
PART 2:
 - Identity unique account names and addresses that are in duplicate account list
 - Isolate all records that have duplicated records 
 - Get the number of times accounts listed in the dataset


"""
#the variables that need to be check for duplicated records
duplicated_name_address=['Account_Name'
                ,'Legal_Street'
                ,'Legal_City'
                ,'Legal_State_Province']


#identify the duplicated records
others['Duplicated'] = others.duplicated(duplicated_name_address)
dealer_vendor['Duplicated'] = dealer_vendor.duplicated(duplicated_name_address)


# export duplicated records for validation
duplicated_others=others.loc[others['Duplicated']==True]
duplicated_dealer_vendor=dealer_vendor.loc[dealer_vendor['Duplicated']==True]


#duplicated_others.to_csv(os.path.join(directory,'duplicated_records_others.csv'), index=False) 
#duplicated_dealer_vendor.to_csv(os.path.join(directory,'duplicated_records_dealer.csv'), index=False)


#get unique duplicate records from others dataset
unique_record_1=duplicated_others.drop_duplicates(duplicated_name_address)
unique_record_1['Tag']='dup_account'
unique_record_others=unique_record_1[['Account_Name'
                                      ,'Legal_Street'
                                      ,'Legal_City'
                                      ,'Legal_State_Province'
                                      ,'Tag']]


#get unique duplicate records from dealer dataset
unique_record_2=duplicated_dealer_vendor.drop_duplicates(duplicated_name_address)
unique_record_2['Tag']='dup_account'
unique_record_dealer=unique_record_2[['Account_Name'
                                      ,'Legal_Street'
                                      ,'Legal_City'
                                      ,'Legal_State_Province'
                                      ,'Tag']]


# Merge original dataset with duplicated account records to get all accounts
#needs to be reviewed
df_others=pd.merge(others,unique_record_others, how='left',on=['Account_Name'\
                                                              ,'Legal_Street'\
                                                              ,'Legal_City'\
                                                              ,'Legal_State_Province'])

df_dealer=pd.merge(dealer_vendor,unique_record_dealer, how='left',on=['Account_Name'\
                                                              ,'Legal_Street'\
                                                              ,'Legal_City'\
                                                              ,'Legal_State_Province'])

# Isolate the all accounts with more than one account_name and addresses
others_to_be_review=df_others[df_others['Tag']=='dup_account']
dealer_to_be_review=df_dealer[df_dealer['Tag']=='dup_account']
"""
****************************************************************************
Accounts need to be reviewed further to see accounts

"""
#Accounts that don't have the same account name and address
further_review_others=df_others[df_others['Tag']!='dup_account']
further_review_dealer=df_dealer[df_dealer['Tag']!='dup_account']

##############################################################################
#further_review_others.to_csv(os.path.join(directory,'non_duplicate_others.csv'), index=False)
#further_review_dealer.to_csv(os.path.join(directory,'non_duplicate_dealer.csv'), index=False)
##############################################################################
others_to_be_review.dtypes

"""
******************************************************************************
"""
#Get number of times accounts listed by Account_Name
number_activity_others=others_to_be_review.groupby(['Account_Name'],as_index=False).agg({'Tag':'count'}) 
number_activity_dealer=dealer_to_be_review.groupby(['Account_Name'],as_index=False).agg({'Tag':'count'}) 


# Drop variables is_duplicated and Tag from the datasets
clean_others=others_to_be_review.drop('Tag', axis=1)
clean_dealer=dealer_to_be_review.drop('Tag', axis=1)



# Merge all duplicated records and the number of duplicated accounts 
number_of_dup_name_others=clean_others.merge(number_activity_others, on='Account_Name', how='left')
number_of_dup_name_dealer=clean_dealer.merge(number_activity_dealer, on='Account_Name', how='left')

"""
*******************************************************************************
PART 3:
 - Separate accounts that listed more than 2 times separately
 - Sort dataset by Account_Name, Customer/Dealer codes, CCAN_MM,CCAN_ST, DUNS and WCIS_ID
 - Export records listed more than 3 times to be review manually
*******************************************************************************

"""
#Separate accounts with more than 2 duplicated accounts
def exception_func(number_of_dup_name_others):
    if number_of_dup_name_others['Tag']>2:
        return 'Exception'
    else:
        return 'Continue'
        
number_of_dup_name_others['Action']=number_of_dup_name_others.apply(exception_func,axis=1)   

def exception_func(number_of_dup_name_dealer):
    if number_of_dup_name_dealer['Tag']>2:
        return 'Exception'
    else:
        return 'Continue'
        
number_of_dup_name_dealer['Action']=number_of_dup_name_dealer.apply(exception_func,axis=1)   


# action_other and action_dealer are to be categorized further
more_than_2_duplicated_others=number_of_dup_name_others[number_of_dup_name_others['Action']=='Exception']
action_others=number_of_dup_name_others[number_of_dup_name_others['Action']!='Exception']
more_than_2_duplicated_dealer=number_of_dup_name_dealer[number_of_dup_name_dealer['Action']=='Exception']
action_dealer=number_of_dup_name_dealer[number_of_dup_name_dealer['Action']!='Exception']

more_than_2_duplicated_dealer.dtypes

#Sort records all records

more_than_2_duplicated_dealer =more_than_2_duplicated_dealer.sort(['Account_Name','Fed_ID___SSN','Dealer_Code','CCAN_(ST)','CCAN_(MM)','DUNS_Number','WCIS_ID'], ascending=[True,False,False,False,False,False,False])
action_dealer =action_dealer.sort(['Account_Name','Fed_ID___SSN','Dealer_Code','CCAN_(ST)','CCAN_(MM)','DUNS_Number','WCIS_ID'], ascending=[True,False,False,False,False,False,False])
more_than_2_duplicated_others =more_than_2_duplicated_others.sort(['Account_Name','Fed_ID___SSN','Customer_Code','Shop_Code','CCAN_(MM)','CCAN_(ST)','DUNS_Number','WCIS_ID'], ascending=[True,False,False,False,False,False,False,False])
action_others =action_others.sort(['Account_Name','Fed_ID___SSN','Customer_Code','Shop_Code','CCAN_(MM)','CCAN_(ST)','DUNS_Number','WCIS_ID'], ascending=[True,False,False,False,False,False,False,False])

"""
#isolate dealer records without dealer code

"""
def dealer_excep_func(more_than_2_duplicated_dealer):
    if (more_than_2_duplicated_dealer['null_Dealer_Code']==False)\
       &(more_than_2_duplicated_dealer['Matching_DCode']==False):
        return 'Survivor'
    else:
        return 'Review'
    
more_than_2_duplicated_dealer['Category']=more_than_2_duplicated_dealer.apply(dealer_excep_func,axis=1)           


# Select field that need to be included in the datasets
keep_field_others=['Created_By'
            ,'Created_Date'
#            ,'Last_Modified_By'
#            ,'Last_Modified_Date'
            ,'Account_Owner'
            ,'LOB'
            ,'Region'
            ,'Division'
            ,'Office'
            ,'Account_Source'
            ,'Account_ID'
            ,'Account_Name'
            ,'Account_Record_Type'
            ,'Legal_Structure'
            ,'Fed_ID___SSN'
            ,'Customer_Code'
            ,'Shop_Code'
            ,'CCAN_(MM)'
            ,'CCAN_(ST)'
            ,'DUNS_Number'
            ,'WCIS_ID'
            ,'Legal_Street'
            ,'Legal_City'
            ,'Legal_State_Province'
            ,'Action']

keep_field_dealer=['Created_By'
            ,'Created_Date'
#            ,'Last_Modified_By'
#            ,'Last_Modified_Date'
            ,'Account_Owner'
            ,'LOB'
            ,'Region'
            ,'Division'
            ,'Office'
            ,'Account_Source'
            ,'Account_ID'
            ,'Account_Name'
            ,'Account_Record_Type'
            ,'Legal_Structure'
            ,'Fed_ID___SSN'
            ,'Dealer_Code'
            ,'CCAN_(MM)'
            ,'CCAN_(ST)'
            ,'DUNS_Number'
            ,'WCIS_ID'
            ,'Legal_Street'
            ,'Legal_City'
            ,'Legal_State_Province'
            ,'Category']



manual_review_more_2_duplicated_others=more_than_2_duplicated_others[keep_field_others]
manual_review_more_2_duplicated_dealer=more_than_2_duplicated_dealer[keep_field_dealer]

# Export records that have more than two duplicated accounts
manual_review_more_2_duplicated_others.to_excel(os.path.join(directory,'manual_review_more_than_2_others.xlsx'), index=False)  
manual_review_more_2_duplicated_dealer.to_excel(os.path.join(directory,'manual_review_more_than_2_dealer.xlsx'), index=False)



"""
*******************************************************************************
PART 4:
 - Check Duplicated Customer/Dealer Code, CCAN_MM,CCAN_ST,DUNS, WCIS_ID and create
 additional variables to be utilized for record labeling
 - Check Null value in Customer/Dealer Code, CCAN_MM,CCAN_ST,DUNS, WCIS_ID and create
 additional variables to be utilized for record labelin
  - Create functions to label records
  - Create a function to isolate expection items to be manually reviewed
  - Isolate records that require manual review
  - Create separate files for accounts labeled as survivor and merge

*******************************************************************************
"""

action_others.dtypes

# create a function to label the records as survivor or merge   
def others_func(action_others):
    if (((action_others['null_Fed_ID']==False)\
       &(action_others['Matching_FedId']==False))\
       &((action_others['Fed_ID___SSN']!=999999999)\
        |(action_others['Fed_ID___SSN']!=123456789)))\
        |(action_others['Legal_Structure']=='Municipality'):
        return 'Survivor'
    elif (action_others['null_Customer_Code']==False)\
       &(action_others['Matching_CCode']==False):
        return 'Survivor'
    elif (action_others['null_Shop_Code']==False)\
       &(action_others['Matching_SCode']==False):
        return 'Survivor'
    elif (action_others['null_CCAN_MM']==False)\
         &(action_others['Matching_CCAN_MM']==False)\
         &(action_others['Duplicated']==False):
        return 'Survivor'
    elif (action_others['null_CCAN_ST']==False)\
         &(action_others['Matching_CCAN_ST']==False)\
         &(action_others['Duplicated']==False):
        return 'Survivor'
    elif (action_others['null_DUNS_Number']==False)\
         &(action_others['Duplicated']==False)\
         &(action_others['Matching_DUNS']==False):
        return 'Survivor'
    elif (action_others['Duplicated']==False)\
         &(action_others['null_WCIS']==False)\
          &(action_others['Matching_WCIS']==False):
        return 'Survivor'   
    elif (action_others['Account_Owner']!='Data Migration')\
         &(action_others['Duplicated']==False):
        return 'Survivor'
    elif (action_others['Account_Owner']=='Data Migration')\
         &(action_others['Duplicated']==False):
        return 'Survivor'        
    else:
        return 'Merge'

# create a function to isolate the records that need to be review manually  
def exception_func(action_others):
    if (action_others['Duplicated']==True):
        if (action_others['null_CCAN_MM']==False)\
           &(action_others['null_CCAN_MM_above']==False)\
           &(action_others['CCAN_(MM)']!=action_others['CCAN_MM_above']):
           return 'Exception'
        elif (action_others['null_CCAN_ST']==False)\
           &(action_others['null_CCAN_ST_above']==False)\
           &(action_others['CCAN_(ST)']!=action_others['CCAN_ST_above']):
           return 'Exception'
        elif (action_others['null_DUNS_Number']==False)\
           &(action_others['DUNS_Number_above']==False)\
           &(action_others['DUNS_Number']!=action_others['DUNS_Number_above']):
           return 'Exception'
    
    

 # create a function to label the records as survivor or merge    
def dealer_func(action_dealer):
    if ((action_dealer['null_Fed_ID']==False)\
       &(action_dealer['Matching_FedId']==False))\
        |(action_dealer['Legal_Structure']=='Municipality'):
        return 'Survivor'
    elif (action_dealer['null_Dealer_Code']==False)\
       &(action_dealer['Matching_DCode']==False):
        return 'Survivor'
    elif (action_dealer['null_CCAN_MM']==False)\
         &(action_dealer['Matching_CCAN_MM']==False)\
         &(action_dealer['Duplicated']==False):
        return 'Survivor'
    elif (action_dealer['null_CCAN_ST']==False)\
         &(action_dealer['Matching_CCAN_ST']==False)\
         &(action_dealer['Duplicated']==False):
        return 'Survivor'
    elif (action_dealer['null_DUNS_Number']==False)\
         &(action_dealer['Duplicated']==False)\
         &(action_dealer['Matching_DUNS']==False):
        return 'Survivor'
    elif (action_dealer['Duplicated']==False)\
         &(action_dealer['null_WCIS']==False)\
          &(action_dealer['Matching_WCIS']==False):
        return 'Survivor'   
    elif (action_dealer['Account_Owner']!='Data Migration')\
         &(action_dealer['Duplicated']==False):
        return 'Survivor'
    else:
        return 'Merge'


#Apply the function to dealer dataset to label them as survivor or merge
action_dealer['Category']=action_dealer.apply(dealer_func,axis=1)    

 
#Apply the function to dealer dataset to label them as survivor or merge         
action_others['Category']=action_others.apply(others_func,axis=1)

# exception function to isolate the records that need to be reviewed manually
action_others['Exception']=action_others.apply(exception_func,axis=1)  

#action_others.to_csv(os.path.join(directory,'test_other_accounts.csv'), index=False)

# Isolate accounts that need to be reviewed manually
exception_2=action_others[action_others['Exception']=='Exception']
exception_account_name=exception_2[['Account_Name']]
exception_account_name['Account_Cat']='exception_accounts'

          
# Separate exception records and labeled records into separate datasets.                    
df_other=pd.merge(action_others,exception_account_name, on='Account_Name', how='left')
others_merge=df_other[df_other['Account_Cat']!='exception_accounts']
others_exception=df_other[df_other['Account_Cat']=='exception_accounts']


# Create final datasets

keep_field_others_1=['Created_By'
            ,'Created_Date'
#            ,'Last_Modified_By'
#            ,'Last_Modified_Date'
            ,'Account_Owner'
            ,'LOB'
            ,'Region'
            ,'Division'
            ,'Office'
            ,'Account_Source'
            ,'Account_ID'
            ,'Account_Name'
            ,'Account_Record_Type'
            ,'Legal_Structure'
            ,'Fed_ID___SSN'
            ,'Customer_Code'
            ,'Shop_Code'
            ,'CCAN_(MM)'
            ,'CCAN_(ST)'
            ,'DUNS_Number'
            ,'WCIS_ID'
            ,'Legal_Street'
            ,'Legal_City'
            ,'Legal_State_Province'
            ,'Category']



other_accounts_to_review_manually=others_exception[keep_field_others_1]
other_accounts_to_review=others_merge[keep_field_others_1]
dealer_accounts_to_review=action_dealer[keep_field_dealer]



# Export data into CSV files for further actions
other_accounts_to_review.to_excel(os.path.join(directory,'other_accounts_to_review.xlsx'), index=False)
dealer_accounts_to_review.to_excel(os.path.join(directory,'dealer_accounts_to_review.xlsx'), index=False)
other_accounts_to_review_manually.to_excel(os.path.join(directory,'manual_review_accounts_others.xlsx'), index=False)

#==============================================================================
# Get fuzzy match from Siva

#==============================================================================

my_file1='Account_De-Dupe Fuzzy Match 12.15.17.xlsx'
fuzzy=pd.read_excel(os.path.join(directory,my_file1),encoding='latin-1')

fuzzy.head()
fuzzy.dtypes


fuzzy['label1']=fuzzy['DUPKEY'].shift(1)
fuzzy['label2']=fuzzy['label1'].shift(1)
fuzzy['label3']=fuzzy['label2'].shift(1)
fuzzy['label4']=fuzzy['label3'].shift(1)


fuzzy.loc[fuzzy['label3'].isnull(),'label3'] = fuzzy['label4']
fuzzy.loc[fuzzy['label2'].isnull(),'label2'] = fuzzy['label3']
fuzzy.loc[fuzzy['label1'].isnull(),'label1'] = fuzzy['label2']

fuzzy.dropna(subset='Owner Full Name',how='all', inplace = True)



fuzzy=fuzzy.sort(['label1','Score1'], ascending=[True,False])

fuzzy['Duplicated']=fuzzy.duplicated(['label1'])




def survivor(df):
    if df['Duplicated']==False:
        return 'Yes'
    else:
        return 'No'
        
fuzzy['Survivor']=fuzzy.apply(survivor,axis=1)  

fuzzy.dtypes
fuzzy.head()
print(fuzzy.columns)


columns=[
        'Id'
        ,'Score1'
        ,'Score2'
        ,'Account Name'
        ,'Legal City'
        ,'Legal State/Province'
        ,'Legal Zip/Postal Code'
        ,'Legal Street'
        ,'Fed ID / SSN'
        ,'CCAN (ST)'
        ,'CCAN (MM)'
        ,'Customer Code'
        ,'Main Market Street'
        ,'Main Market City'
        ,'Main Market State/Province'
        ,'Main Market Zip/Postal Code'
        ,'Owner Full Name'
        ,'CreatedBy Full Name'
        ,'label1'
        ,'Survivor'
        ]


fuzzy=fuzzy[columns]


accounts=df1[[
            'Account_ID'
            ,'DUNS_Number'
            ,'WCIS_ID'
            ,'WCIS_DUNS_Number'
            ]]




fuzzy['Id']=fuzzy['Id'].str[:15]

fuzzy_match=pd.merge(fuzzy,accounts, how='left',left_on=['Id'],right_on=['Account_ID'])

print(fuzzy_match.columns)

columns=[
        'Account_ID'
        ,'Score1'
        ,'Account Name'
        ,'Legal Street'
        ,'Legal City'
        ,'Legal State/Province'
        ,'Legal Zip/Postal Code'
        ,'Fed ID / SSN'
        ,'CCAN (ST)'
        ,'CCAN (MM)'
        ,'Customer Code'
        ,'DUNS_Number'
        ,'WCIS_ID'
        ,'WCIS_DUNS_Number'
        ,'Main Market Street'
        ,'Main Market City'
        ,'Main Market State/Province'
        ,'Main Market Zip/Postal Code'
        ,'Owner Full Name'
        ,'CreatedBy Full Name'
        ,'Survivor'
        ,'label1'
        ]

fuzzy_match=fuzzy_match[columns]


fuzzy_match=fuzzy_match[fuzzy_match['Score1'].isnull()==False]
fuzzy_match=fuzzy_match.sort(['label1','Score1'], ascending=[True,False])




fuzzy_match.to_excel(os.path.join(directory,'fuzzy_match_review_1.02.18.xlsx'), index=False)

