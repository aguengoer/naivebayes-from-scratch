import pandas as pd

df = pd.read_csv('naives_bayes_data.csv')
print(df.to_string())
###calculate all likelihoods for non spam
non_spam_dict = {}
df_non_spam = df.loc[df['spam'] == 0]
sum_of_all_words_in_non_spam = 0
total_count_non_spam = df_non_spam.shape[0]

for c in df.columns:
    if c not in "spam":
        non_spam_dict[c] = sum(df_non_spam[c].values)
        sum_of_all_words_in_non_spam = sum_of_all_words_in_non_spam + sum(df_non_spam[c].values)

likelihood_for_non_spam = {}

for key in non_spam_dict:
    likelihood_for_non_spam[key] = non_spam_dict[key] / sum_of_all_words_in_non_spam

###calculate all likelihoods for spam
spam_dict = {}
df_spam = df.loc[df['spam'] == 1]
sum_of_all_words_in_spam = 0
total_count_spam = df_spam.shape[0]

for c in df.columns:
    if c not in "spam":
        spam_dict[c] = sum(df_spam[c].values)
        sum_of_all_words_in_spam = sum_of_all_words_in_spam + sum(df_spam[c].values)

likelihood_for_spam = {}
for key in spam_dict:
    likelihood_for_spam[key] = spam_dict[key] / sum_of_all_words_in_spam

print("All likelihoods for non spam: ", likelihood_for_non_spam)
print("All likelihoods for spam: ", likelihood_for_spam)
print('\n')

test_data = ['dear pizza friend', 'free bitcoin', 'today code-review', 'free pizza', 'free code-review', 'free today']

for td in test_data:
    splitted_td = td.split(' ')
    non_spam_multiplied_value = 1
    spam_multiplied_value = 1
    print('test data: ', td)
    for s_td in splitted_td:
        non_spam_multiplied_value = likelihood_for_non_spam[s_td] * non_spam_multiplied_value
        spam_multiplied_value = likelihood_for_spam[s_td] * spam_multiplied_value
    non_spam_multiplied_value = non_spam_multiplied_value * total_count_spam / df.shape[0]
    spam_multiplied_value = spam_multiplied_value * total_count_spam / df.shape[0]
    print('total likelihood for non spam: ', non_spam_multiplied_value)
    print('total likelihood for spam: ', spam_multiplied_value)
    if non_spam_multiplied_value > spam_multiplied_value:
        print('Not Spam Email')
    else:
        print('Spam Email')
    print('\n')

# Laplace/Additive Smoothing --> prevent 0 likelihoods --> should do it for all data!
# for key in non_spam_dict:
#     if non_spam_dict[key] == 0:
#         for k in non_spam_dict:
#             non_spam_dict[k] = non_spam_dict[k] + 1
#             sum_of_all_words_in_non_spam = sum_of_all_words_in_non_spam + 1
