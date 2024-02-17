# Building a Spam Filter with Naive Bayes

In this project, we're going to build a spam filter for SMS messages using the multinomial Naive Bayes algorithm. Our goal is to write a program that classifies new messages with an accuracy greater than 80% — so we expect that more than 80% of the new messages will be classified correctly as spam or ham (non-spam).

To train the algorithm, we'll use a dataset of 5,572 SMS messages that are already classified by humans. The dataset was put together by Tiago A. Almeida and José María Gómez Hidalgo, and it can be downloaded from the The UCI Machine Learning Repository.


```python
import pandas as pd
```


```python
sms_spam = pd.read_csv(r"C:\Users\BUSINESS ANALYST\OneDrive - BBOXX\Documents\Project_files\SMSSpamCollection", sep= '\t', header = None, names = ['Label','Sms'])

print(sms_spam.shape)
print(sms_spam.head())

```

    (5572, 2)
      Label                                                Sms
    0   ham  Go until jurong point, crazy.. Available only ...
    1   ham                      Ok lar... Joking wif u oni...
    2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3   ham  U dun say so early hor... U c already then say...
    4   ham  Nah I don't think he goes to usf, he lives aro...
    


```python
#We find that 13% of the SMSs have been labelled as Spam
sms_spam['Label'].value_counts(normalize = True)
```




    Label
    ham     0.865937
    spam    0.134063
    Name: proportion, dtype: float64



## Training The Test Set

We're going to keep 80% of our dataset for training, and 20% for testing (we want to train the algorithm on as much data as possible, but we also want to have enough test data). The dataset has 5,572 messages, which means that:

 - The training set will have 4,458 messages (about 80% of the dataset).
 - The test set will have 1,114 messages (about 20% of the dataset).


```python
# We first need to tandomize the dataset (frac = 1 is to randomize & random_state is to make the result reproducible)

data_randomized = sms_spam.sample(frac = 1, random_state = 1)

# give 80% to training set
training_test_index= round(len(data_randomized)* 0.8)


# Splitting the two datasets (drop = true ensures that we dont carry the previous dfs index to the new data)

training_set = data_randomized[:training_test_index].reset_index(drop = True)

test_set = data_randomized[training_test_index:].reset_index(drop = True)

print(training_set.shape)
print(test_set.shape)

# to check the distribution of the ham & spam in the data sets

print(training_set['Label'].value_counts(normalize = True))

print(test_set['Label'].value_counts(normalize = True))

```

    (4458, 2)
    (1114, 2)
    Label
    ham     0.86541
    spam    0.13459
    Name: proportion, dtype: float64
    Label
    ham     0.868043
    spam    0.131957
    Name: proportion, dtype: float64
    

## Data Cleaning

The goal is to have the data to like the below format

![image.png](attachment:image.png)


```python
# to recap. Training set before cleaning
training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Yep, by the pretty sculpture</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Yes, princess. Are you going to make me moan?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>Welp apparently he retired</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>Havent.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>I forgot 2 ask ü all smth.. There's a card on ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Cleaning process
# we use the '\W' to replace anything that is not a letter, digit, or underscore)
training_set['Sms'] = training_set['Sms'].str.replace('\W','')
training_set['Sms'] = training_set['Sms'].str.replace('?','')
training_set['Sms'] = training_set['Sms'].str.replace(',','')
training_set['Sms'] = training_set['Sms'].str.replace('.','')
training_set['Sms'] = training_set['Sms'].str.lower()
training_set.head()


# # Cleaning process
# # Replace anything that is not a letter, digit, or underscore, including '?'
# training_set['Sms'] = training_set['Sms'].str.replace('[^\w\s]', ' ')
# # Convert to lowercase
# training_set['Sms'] = training_set['Sms'].str.lower()
# # Display the first few rows
# training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>yep by the pretty sculpture</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>yes princess are you going to make me moan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>welp apparently he retired</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>havent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>i forgot 2 ask ü all smth there's a card on da...</td>
    </tr>
  </tbody>
</table>
</div>



## Creating a vocabulary

- This is creating a set of all unique words in our training set



```python
training_set['Sms'] = training_set['Sms'].str.split()

vocabulary = []

for sms in training_set['Sms']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))   


len(vocabulary)
```




    9414




```python
vocabulary
```




    ['closingdate04/09/02',
     '82468',
     'aid',
     '09050280520',
     'claim',
     'and',
     'discuss',
     'la1',
     'best1',
     '(quizclub',
     '83355',
     'unni',
     'radio',
     'permissions',
     '09066362206',
     'realise',
     '3wife:',
     'tahan',
     'icon',
     'salesman',
     'replybe',
     'bpo',
     'bfore',
     'sfrom',
     'action',
     'abnormally',
     'planettalkinstantcom',
     'what!',
     'staying',
     'campus',
     'remb',
     "basket's",
     'ok!',
     'oyster',
     '08718726970',
     'jstfrnd',
     'wordnot',
     'electricity',
     'sunshine!',
     'gimmi',
     'hello-/@drivby-:0quit',
     'endowed',
     'neither',
     'tmw',
     'mymoby',
     "don't4get2text",
     'maintaining',
     'strong:)',
     '0906346330',
     'lennon',
     'issue',
     'semester',
     'other!',
     'splat!',
     'grateful',
     '"its',
     'patty',
     'calls150ppm',
     'syllabus',
     '8552',
     'adds',
     'yeh',
     'competition"',
     'clarify',
     'fren',
     'wherevr',
     'hols',
     'internet',
     'sticky',
     't&cs/stop',
     'picture',
     'walked',
     'tomo',
     'hearing',
     'sorry-i',
     'one',
     'jo',
     'madam',
     'eg',
     'abt',
     'wearing',
     '5000',
     'apes',
     'shiny',
     'married',
     'shakara',
     'no-165',
     'concern',
     'passion',
     'badrith',
     '"smokes',
     '1da',
     'violet',
     'croydon',
     'manage',
     'gotmarried',
     'favorite',
     'marry',
     'agidhane',
     'gbp',
     'cine',
     'na',
     'castor',
     'arise',
     'toyota',
     '5gardener:',
     'h&m',
     '09066368327',
     'randomlly',
     'bk',
     'dead"',
     'rearrange',
     'newest',
     'apt',
     'oil',
     'uawakefeellikw',
     'correctly',
     'franyxxxxx',
     'hrs',
     'nw',
     '$1',
     'borrow',
     'novelty',
     'wined',
     '4041',
     'bffs',
     'getting',
     '4goten',
     'attitude',
     'camry',
     'moves',
     'blessing',
     'docs',
     '3510i',
     'vldo',
     'this',
     'da:)where',
     'got',
     'whilltake',
     '09050000878',
     'atten',
     'lines',
     'obese',
     'this:)don',
     'spageddies',
     '69669',
     'yijue',
     'yeah',
     'headset',
     'than',
     'smoking',
     'stereo',
     '"oh"',
     'suzy',
     'sure!',
     'credit',
     'daalways',
     'truekdo',
     'cometil',
     'everythin',
     'bored!',
     'itried2tell',
     'dao',
     'cereals',
     'blonde',
     'exmpel:',
     'especially',
     'part',
     'no1',
     'answers',
     'abroad',
     '09053750005',
     'yo',
     'cali',
     'maximize',
     'polyc',
     'talent',
     'ship',
     'dang!',
     'lapdancer',
     'furniture',
     'bids',
     'violated',
     'outages',
     '09050002311',
     '07123456789',
     '&lt;)',
     'emergency',
     'phone!',
     '0870241182716',
     'goa',
     'applyed',
     'girls',
     "aren't",
     'shoul',
     'ls15hb',
     'nice!',
     'men',
     'love!!',
     'excellent!',
     'smokes',
     'through',
     '-sub',
     'twice-',
     'chatter!',
     'bring',
     '89034',
     'typelyk',
     'mark',
     'lucky',
     'somerset',
     'nag',
     'shivratri',
     'confirm',
     'stand',
     'ondu',
     'sake!',
     "y'day",
     'beside',
     'name',
     'dogging',
     'goin',
     'reaching',
     'hopeafternoon',
     '£3',
     'dasara',
     'cro1327',
     'soundtrack',
     'sha',
     'wa',
     'slave!',
     'polyphonic',
     'stamped',
     'barrel',
     'lololo',
     'window',
     'hairdressers',
     'lim',
     'oredi',
     'kanowhr',
     'rise',
     'outfor',
     'kanoanyway',
     'orchard',
     'then',
     'kusruthi',
     'bus',
     'express',
     'john',
     'odalebeku:',
     'story:-',
     'vale',
     'day!2find',
     'care!',
     'aids',
     'done/want',
     'timing',
     'dammit!!',
     'tm',
     'nojst',
     'kids',
     'distract',
     'perfume',
     'study',
     'final',
     '8883',
     'portal',
     'ü',
     'working',
     'nokia',
     '(flights',
     'under',
     '8neighbour:',
     'il',
     'wah',
     'is:',
     't&cs',
     'bangb',
     'without',
     'inour',
     "said''",
     'remembr',
     '420',
     'lie',
     'luv',
     'datz',
     'queries',
     'zhong',
     'sp',
     'yowifes',
     ':',
     'weight!',
     'sp:rwm',
     'you',
     'melnite',
     'convince',
     'scratching',
     'thy',
     '*phews*',
     'scotch',
     'hey',
     '09094646631',
     'off:)',
     'darren',
     'fatty',
     'feathery',
     'blackberry',
     'country',
     'unsubscribed',
     'thousands',
     'helloooo',
     'havebeen',
     'kegger)',
     "b'coz",
     'forum',
     'wap',
     'sunday:)',
     'http://wwwbubbletextcom',
     '4-7/12',
     'paranoid',
     'singapore',
     'splash',
     '4brekkie!',
     'stability',
     'formsdon',
     'massive',
     'twelve!',
     'contract',
     'tip',
     'finding',
     'aa',
     'lots',
     '61610',
     'unrecognized',
     '09066660100',
     'afew',
     'tired',
     'killed',
     '4u',
     'dark',
     'hor',
     'brdget',
     "mine's",
     'sory',
     '9ae',
     'literally',
     '(book',
     'holby!',
     'shoranur',
     'call2optout/lf56',
     'light!',
     'chat',
     'uks',
     'fresh',
     "sms'd",
     'wmlid=820554ad0a1705572711&first=true¡c',
     ':getzedcouk',
     'slow',
     'ic',
     'sinco',
     'drpd',
     '150p/mtmsg',
     '9153',
     'selflessness',
     'lay',
     'shove',
     'count',
     'on)',
     'hmmmkbut',
     '08718726978',
     'usps',
     'thenwill',
     "treatin'",
     '£350',
     'collect',
     'ipod!',
     '**free',
     'yourinclusive',
     'neshanthtel',
     'shocking',
     '=',
     'swear',
     'knows',
     'satjust',
     'jokin',
     'quiteamuzing',
     'nit',
     'hava',
     'reslove',
     'hesitate',
     'dough',
     'desparately',
     'sugababes',
     'gopalettan',
     'askin',
     'flaky',
     'polys:',
     'resubbing',
     '500',
     'inform',
     'roommate',
     'dresser',
     'accent',
     '08712402972',
     'teethif',
     'bud',
     'pdate_now',
     '18+only',
     '£900',
     'lily',
     'cheating',
     'upgrading',
     'dismissial',
     'duo',
     'nowadays:)lot',
     '08000930705',
     'friend-of-a-friend',
     'finishd',
     'attempt',
     'hello',
     'hundred',
     'am',
     'french',
     'bedreal',
     'prey',
     '08714712412',
     'ni8;-)',
     'wamma',
     'coco',
     'radiator!',
     'lul',
     'parantella',
     'promise',
     'expensive',
     'tc',
     "she's",
     '"mix"',
     'jerry',
     'agesring',
     's:)8',
     'vday',
     'hun-onbus',
     'quarter',
     'course:)',
     'rg21',
     'funny',
     '"song',
     'successfully',
     '08000407165',
     'glands',
     'missed',
     'from',
     '50',
     'thinkthis',
     '5th',
     'msgs:d;):',
     'transferred',
     'stuff42moro',
     'convincingjust',
     'oh:)as',
     'ability',
     'cc',
     'sleepin',
     'prsn',
     'went',
     'corporation',
     'cooked',
     '2p',
     'owo',
     'box334sk38ch',
     'gud',
     'monthnot',
     'sambarlife',
     'most',
     '2godid',
     '*9',
     'inlude',
     'suprman',
     'kodthini!',
     'guide',
     'balloon!',
     'penis',
     'generally',
     'entertain',
     'conditionand',
     '24/10/04',
     'noooooooo',
     '£100000',
     'custom',
     'msg/subscription',
     'that!',
     'd=',
     'norcorp',
     'listening2the',
     '2px',
     'analysis',
     'apartment',
     'answr',
     'tablet',
     'http://tms',
     'scammers',
     'firmware',
     'computer',
     'christians',
     'a£50',
     'chances',
     'callcoz',
     '08715203656',
     'msgs@150p',
     'if/when/how',
     'whatever',
     '*loving',
     'costa',
     'yetunde',
     'mokka',
     'enter',
     'duchess',
     '08712466669',
     'diwali',
     'locations',
     'flowers',
     'ifink',
     'refilled',
     'organise',
     'everything',
     'jay',
     'savings',
     'hg/suite342/2lands',
     'sooner',
     'cw25wx',
     'emotion',
     'b',
     'freaky',
     'petrol-rs',
     'available',
     'secrets',
     'bluetoothhdset',
     'throws',
     '4u!:',
     'invited',
     'true"',
     "dsn't",
     'k:)do',
     'includes',
     '69888!',
     'deposit',
     'beforehand',
     'switch',
     'wishin',
     'raining',
     '09063458130',
     '4',
     '5min',
     'chase',
     'recieve',
     'aldrine',
     'antibiotic',
     'threw',
     'fo',
     'nottel',
     'farm',
     'allowed',
     "cali's",
     'apologise',
     'c!',
     'payed',
     '6pm:-)',
     'petexxx',
     'cheers',
     'shop--the',
     'korli',
     '08718723815',
     'claimcode',
     'followed',
     'smoothly',
     'renewed',
     'sometme',
     'ahsen',
     '87077',
     'gaps',
     'cheyyamoand',
     'macha',
     'exchanged',
     'tasts',
     'networking',
     'mids',
     'win',
     'salmon',
     'wishes',
     'doubt',
     'chance!',
     'heal',
     '861',
     'illness',
     'ache',
     'mu',
     'boy:',
     'though',
     'avin',
     '60p/min',
     'ps',
     'fffff',
     'm100',
     'loss',
     'fainting',
     'westshore',
     'pan',
     'census',
     'no:-)i',
     'madstini',
     'learn',
     'eek',
     'misundrstud',
     'wwwwin-82050couk',
     'blanked',
     'wkend',
     'passionate',
     '0870737910216yrs',
     'chain',
     'armenia',
     'aco&entry41',
     'little',
     'ringtone¡',
     'trained',
     'sp:tyrone',
     '81151',
     'chikku:-):-db-)',
     "c's",
     'vivek:)i',
     'lst',
     "''",
     'deal:-)',
     'batch',
     'wet',
     'somewheresomeone',
     'club:',
     'google',
     'snake',
     'vote',
     '£800',
     'no!listened2the',
     'morning!',
     '29/10/0',
     'card!',
     'common',
     'kept',
     'nowcan',
     'whassup',
     'ryan',
     'abiola',
     'grazed',
     'audition',
     'drive"',
     '150p/mtmsgrcvd18+',
     '(not',
     'purity',
     'u:-)',
     'smsshsexnetun',
     'years!',
     'help:08712400602450p',
     'anyplaces',
     'mahal',
     'je',
     'penny',
     'reduce',
     'coccooning',
     'pobox365o4w45wq',
     'named',
     'cheaper',
     'doublemins',
     'crazy',
     'nattil',
     'jordan!txt',
     'gran',
     'keeps',
     'vegas',
     'mmmmmmm',
     'elaya',
     'prizes',
     'allahmeet',
     "how're",
     'patent',
     'mate',
     'tag',
     'psychic',
     'numberpls',
     'audrie',
     'nicenicehow',
     'contact',
     'start',
     'flag',
     'reality',
     'alternativehope',
     'kisi',
     'exhaust',
     'conected',
     'people',
     'gifts!!',
     '3030',
     'well!',
     'lifpartnr',
     'gal',
     'kent',
     'lotr',
     'stated',
     'screaming',
     'appointments',
     'voicemail',
     'right',
     'hol',
     'woman',
     'cook',
     'temp',
     'says',
     'you:)k:)where',
     'profile',
     '430',
     'dorm',
     'system',
     'toothpaste',
     'sum',
     'call:',
     'fault&al',
     'decisions',
     'saeed',
     '08718711108',
     'befor',
     'still',
     'muht',
     'kanoil',
     'whether',
     'latr',
     'thinkin',
     '<ukp>2000',
     'software',
     'operate',
     'none',
     'broken',
     '3x£150pw',
     'glo',
     '3680',
     'weddin',
     'wwwdbuknet',
     'winner!',
     '*kiss*',
     'peace',
     "unicef's",
     'nobut',
     'addamsfa',
     'yours!',
     'added',
     'onam',
     'own',
     'breathe',
     '505060',
     'comprehensive',
     'cuddling',
     'needle',
     'jocks',
     'announcement',
     'anniversary',
     'nokia6650',
     'pushbutton',
     'logos+musicnews!',
     'sorryin',
     'weekly',
     'anybody',
     'golden',
     'tactless',
     'started',
     'makin',
     'talking',
     'properly',
     'possibility',
     'bruce',
     'colleg',
     'foregate',
     '08714712394',
     '08001950382',
     'treats',
     'da:)how',
     'jackpot!',
     'bein',
     '50p',
     '087018728737',
     'dearly',
     'moving',
     '2wks',
     'keyword',
     'ts&cs',
     'usualiam',
     'leave',
     'mins',
     'se',
     'anytime',
     '£500',
     'man!',
     'free>ringtone!reply',
     'burgundy',
     'lehhaha',
     'nick',
     'bedroom',
     'convincing',
     'itmail',
     'shanilrakhesh',
     'algebra',
     'havin',
     'free-nokia',
     'tallent',
     'payments',
     'needed',
     'dramatic',
     'im',
     'keeping',
     'toleratbcs',
     'actor',
     'macho',
     'advance',
     'bc',
     'lt',
     'actually',
     'teacher',
     'raping',
     'shipped',
     'happier',
     'minnaminunginte',
     'checked',
     'availablethey',
     'bao',
     'connection',
     'style',
     'ms',
     'closeby',
     'hours',
     'india:-):',
     '4d',
     'reacting',
     'timings',
     'asthma',
     'i\x92m',
     'bluff',
     'coach',
     'eyes',
     'but',
     'stranger',
     'unhappiness!',
     'laidwant',
     'die',
     'highest',
     'auntie',
     'mobileupd8',
     'slap',
     'several',
     'dudes',
     'affairs',
     'txt>',
     'australia',
     'stressfull',
     'smarter',
     'modl',
     'poo',
     '09071512433',
     'hence',
     'once',
     'movie',
     'max',
     'aeroplane',
     '07099833605',
     'tui',
     'u2moro',
     "1000's",
     'waqt',
     'deleted',
     'same',
     'practising',
     'yoga',
     'mobs',
     'really',
     'creative',
     'big',
     'screen',
     'moments',
     '9t',
     't-ruthful',
     'chillin',
     'bx420-ip4-5we',
     'pocketbabecouk',
     '18+6*£150(morefrmmob',
     'icky',
     'yifeng',
     'w1t1jy',
     'director',
     'thriller',
     '3-u',
     'arms',
     'our',
     "er'ything!",
     'vasai',
     'treat',
     'woulda',
     'words-',
     'dollars',
     '2optout/d3wv',
     'ringtone!from:',
     'special-call',
     'bsnl',
     'incident',
     'today',
     'ctter',
     '08000776320',
     '5+-',
     'youmoney',
     'hello"',
     'the',
     'heat',
     'spoon',
     'premium',
     'lifebook',
     '89938',
     'property',
     'combine',
     '***************',
     'spare',
     'created',
     'length',
     'bunkers',
     'feels',
     'scarcasim',
     'snow',
     'euro2004',
     '0845',
     'raj',
     'thursday',
     'although',
     'starti',
     '"valued',
     'enjoy!',
     'deyhope',
     'halla',
     "i'ma",
     '6pm',
     'invest',
     'ice',
     'egg',
     'diamond',
     'carolina',
     'that\x92s',
     'swimming',
     '1148',
     'avoid',
     'mnth',
     '62735=£450',
     'humanities',
     'gloucesterroad',
     'nichols',
     'chicken',
     'london',
     'hmph',
     'systems',
     'f=',
     'beeen',
     'lined',
     'official',
     'jan',
     'hun!love',
     'parts',
     'txtauction!',
     'affectionate',
     'cheat',
     'downstem',
     'slots',
     'yunny',
     'nope',
     'gotany',
     'inc',
     'posible',
     'x49your',
     'okok',
     'for£38',
     'nigeria',
     '118p/msg',
     '1er',
     'galileo',
     'cheese',
     'loverboy!',
     'flavour',
     'reformat',
     'free>ringtone!',
     'who',
     'newsby',
     'goverment',
     'pansy!',
     'liver',
     'time',
     '09066382422',
     '09701213186',
     'pray',
     'ip',
     'sayhey!',
     'comp',
     ...]



# Creating the final Training set


```python
# word_counts_per_sms = {}  # Initialize an empty dictionary
# for unique_word in vocabulary:
#     word_counts_per_sms[unique_word] = [0] * len(training_set['Sms'])

word_counts_per_sms = {unique_word: [0] * len(training_set['Sms']) for unique_word in vocabulary}

for index, sms in enumerate (training_set['Sms']):
    for word in sms:
        word_counts_per_sms[word][index] +=1

```


```python
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.shape
```




    (4458, 9414)




```python
training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Sms</th>
      <th>closingdate04/09/02</th>
      <th>82468</th>
      <th>aid</th>
      <th>09050280520</th>
      <th>claim</th>
      <th>and</th>
      <th>discuss</th>
      <th>la1</th>
      <th>...</th>
      <th>achanammarakheshqatar</th>
      <th>masteriastering</th>
      <th>10:30</th>
      <th>no:81151</th>
      <th>pple</th>
      <th>946</th>
      <th>weed-deficient</th>
      <th>could</th>
      <th>prove</th>
      <th>jobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>[yep, by, the, pretty, sculpture]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>[yes, princess, are, you, going, to, make, me,...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>[welp, apparently, he, retired]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>[havent]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>[i, forgot, 2, ask, ü, all, smth, there's, a, ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9416 columns</p>
</div>




```python
training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>[yep, by, the, pretty, sculpture]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>[yes, princess, are, you, going, to, make, me,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>[welp, apparently, he, retired]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>[havent]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>[i, forgot, 2, ask, ü, all, smth, there's, a, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
training_set['Sms'][:10]
```




    0                    [yep, by, the, pretty, sculpture]
    1    [yes, princess, are, you, going, to, make, me,...
    2                      [welp, apparently, he, retired]
    3                                             [havent]
    4    [i, forgot, 2, ask, ü, all, smth, there's, a, ...
    5    [ok, i, thk, i, got, it, then, u, wan, me, 2, ...
    6    [i, want, kfc, its, tuesday, only, buy, 2, mea...
    7                    [no, dear, i, was, sleeping, :-p]
    8                        [ok, pa, nothing, problem:-)]
    9                  [ill, be, there, on, &lt;#&gt;, ok]
    Name: Sms, dtype: object




```python
for sms in training_set['Sms'][:3]:
    print(sms)
```

    ['yep', 'by', 'the', 'pretty', 'sculpture']
    ['yes', 'princess', 'are', 'you', 'going', 'to', 'make', 'me', 'moan']
    ['welp', 'apparently', 'he', 'retired']
    


```python
training_set['Sms'][:3
```
