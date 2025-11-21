import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('trainingsset-including-seed.csv')

########################################################################################
#                          1. Wie groß ist der Datensatz
########################################################################################
                    #  <class 'pandas.core.frame.DataFrame'>
                    # RangeIndex: 8270 entries, 0 to 8269
                    # Data columns (total 6 columns):
                    #  #   Column        Non-Null Count  Dtype
                    # ---  ------        --------------  -----
                    #  0   question      8270 non-null   object
                    #  1   label         8270 non-null   object
                    #  2   severity      8002 non-null   object
                    #  3   category      1698 non-null   object
                    #  4   label_binary  8270 non-null   bool
                    #  5   qid           0 non-null      float64
                    # dtypes: bool(1), float64(1), object(4)
                    # memory usage: 331.3+ KB
                    # Info: None

########################################################################################
#                   2. Wie lange sind die Anfragen im Durchschnitt
########################################################################################
df['clean_questions'] = df['question'].str.replace(r'#####\s*\r?\n', '', regex=True)
df['length'] = df['clean_questions'].str.len() # --> 206.65392986698913

sns.kdeplot(df['length'])

plt.tight_layout(rect=[0, 0, 0.90, 0.95])
plt.xlim(0, 800)
plt.title('Wie lange sind die Anfragen im Durchschnitt')
plt.xlabel('Lange')
plt.ylabel('Dichte')
plt.annotate(
            'durchschnittliche Länge der Anfragen = 206.6',
            xy=(210, 0.0023),
            xytext=(320, 0.0015),
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            fontsize=9,
            color='blue'
            )
plt.scatter(210, 0.0023, color='black')
plt.show()

########################################################################################
#           3. Unterscheiden sich die Anfragen in der Länge bezogen je nach Label?
########################################################################################
true_false = df.groupby('label_binary')
mean_length = true_false['length'].mean()
false_mean = true_false.get_group(False)['length'].mean() # -> 169.6
true_mean = true_false.get_group(True)['length'].mean() # -> 243.6


palette = {False: 'red', True: 'green'}
sns.kdeplot(data=df, x='length', hue='label_binary', palette=palette)


plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.xlim(0, 800)
plt.xlabel('Lange')
plt.ylabel('Dichte')
plt.title('Unterscheiden sich die Anfragen in der Länge bezogen je nach Label?')
plt.annotate(
            text='Durchschnittliche Länge der Anfragen "True" = 243.6 ',
            xy=(243.6, 0.00088),
            xytext=(280, 0.00130),
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            fontsize=9,
            color='black'
            )
plt.scatter(243.6, 0.00088, color='black')
plt.annotate(
            text='Durchschnittliche Länge der Anfragen "True" = 169.6 ',
            xy=(171, 0.00150),
            xytext=(280, 0.00200),
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            fontsize=9,
            color='black'
            )
plt.scatter(171, 0.00150, color='black')

plt.show()

########################################################################################
#   4. Gibt es bestimmte Keywords zu harmlosen Themen, zB Bank, Kreditkarte, Mail etc,
#                   die unterschiedlich häufig sind je nach Label?
########################################################################################

key_words = ['Bank', 'E-Mail', 'Kreditkarte', 'Überweisung']
key_words_counts = {}

for key_word in key_words:
    df[key_word] = df['clean_questions'].str.contains(key_word, case=False, na=False)

for key_word in key_words:
    counts = df.groupby('label_binary')[key_word].sum()
    key_words_counts[key_word] = counts

key_words_counts_df = pd.DataFrame(key_words_counts)

key_words_counts_df.plot(kind='bar')
plt.xticks(rotation=0)
plt.show()