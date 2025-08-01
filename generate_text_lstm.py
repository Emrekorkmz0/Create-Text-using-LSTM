
# %%
#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  create dataset
dataset = [
    "Kitap okumak beni çok mutlu ediyor.",
    "Bugün hava çok güzel.",
    "Sabah kahvemi içmeden güne başlayamıyorum.",
    "Yolda çok fazla trafik vardı, işe geç kaldım.",
    "Yeni diziyi izledin mi? Harikaydı!",
    "Telefonum yine bozuldu, çok sinirliyim.",
    "Yürüyüş yapmak zihnimi dinlendiriyor.",
    "Bugün hiç keyfim yoktu.",
    "Ödevleri bitirmek saatler sürdü.",
    "Deniz kenarında oturmak çok huzurlu.",
    "Müzik dinlemek ruhuma iyi geliyor.",
    "Dün akşam yemeği çok lezzetliydi.",
    "Sabah erken kalkmak bana zor geliyor.",
    "Tatil planı yaparken çok heyecanlandım.",
    "Kalabalık yerlerden pek hoşlanmıyorum.",
    "Kitapçıda saatlerce vakit geçirdim.",
    "Yeni telefonumun kamerası harika.",
    "Film çok uzundu ama sonu güzeldi.",
    "Bugün spora gitmek istemiyorum.",
    "Kütüphane sessiz ve huzurluydu.",
    "Kahve içmeden ders çalışamıyorum.",
    "Arkadaşlarımla pikniğe gittik.",
    "Gece geç saatlere kadar çalıştım.",
    "Toplantı beklediğimden uzun sürdü.",
    "Bilgisayarım aniden kapandı.",
    "Sınavdan düşük not aldım, moralim bozuk.",
    "Hafta sonu denize gitmeyi planlıyorum.",
    "Yeni bir şeyler öğrenmek beni mutlu ediyor.",
    "Köpeğimle parkta yürüyüş yaptım.",
    "Bugün kendimi enerjik hissediyorum.",
    "Yağmur yağınca camdan dışarıyı izlemeyi seviyorum.",
    "Ev işleri hiç bitmiyor.",
    "Bu sabah kahvaltı yapmayı unuttum.",
    "Ders çalışırken dikkatim çabuk dağılıyor.",
    "Oyun oynamak bana çok keyif veriyor.",
    "Kardeşimle kavga ettim, biraz üzgünüm.",
    "Güneşli havalarda yürümek çok güzel.",
    "Filmdeki oyunculuklar gerçekten etkileyiciydi.",
    "Bugün ofiste çok yoğunduk.",
    "Yeni ayakkabılarım ayağımı vurdu.",
    "Kalemimi evde unuttuğum için derste not alamadım.",
    "Yemekten sonra biraz kestirdim.",
    "Radyoda eski bir şarkı çaldı, duygulandım.",
    "Yemek yapmak bazen terapi gibi geliyor.",
    "Uçağım rötar yaptı, uzun süre bekledim.",
    "Arkadaşımın doğum günü çok eğlenceliydi.",
    "Bugün kitapçıda indirim vardı.",
    "Yeni bir dil öğrenmeye başladım.",
    "Markette her şey çok pahalıydı.",
    "Dün gece rüyamda eski bir arkadaşımı gördüm.",
    "Bu sabah erkenden yürüyüşe çıktım.",
    "Bahar geldi, her yer çiçek açtı.",
    "Karnım çok aç, ne yesem bilemiyorum.",
    "Şemsiyemi unuttuğum için ıslandım.",
    "Telefonumun şarjı çok çabuk bitiyor.",
    "Akşam yemeği için ne pişireceğimi bilmiyorum.",
    "Bugün hiç mesaj almadım.",
    "Kütüphanede çok güzel bir kitap buldum.",
    "Yeni saç modelim çok beğenildi.",
    "Sınav sonucum beklediğimden iyiydi.",
    "Akşam dışarı çıkmak istiyorum.",
    "Film çok güzeldi, tekrar izlemek istiyorum.",
    "Kütüphaneye gitmek bana huzur veriyor.",
    "Hava soğuk ama güneşliydi.",
    "Kamp yapmayı çok özlemişim.",
    "Tatilde bol bol kitap okudum.",
    "Sabah alarmı duymamışım, geç kaldım.",
    "Bugün işler hiç yolunda gitmedi.",
    "Spor salonunda kendimi çok iyi hissediyorum.",
    "Kedim sabah beni erkenden uyandırdı.",
    "Arkadaşım bana sürpriz yaptı, çok mutlu oldum.",
    "Ders çalışırken müzik dinlemeyi seviyorum.",
    "Yeni laptopum çok hızlı.",
    "Bugün kahvaltı çok keyifliydi.",
    "Çiçek sulamayı unuttum, solmuşlar.",
    "Telefonumda yer kalmadı.",
    "Sınav çok zordu, moralim bozuldu.",
    "Yürüyüş yaparken podcast dinliyorum.",
    "Tatlı yapmayı yeni öğrendim.",
    "Hava yağmurlu olduğu için evde kaldım.",
    "Kütüphane çok kalabalıktı, yer bulamadım.",
    "Sabah koşusu bana iyi geliyor.",
    "Bugün annemle alışverişe çıktık.",
    "Kahvemi döktüm, masam kirlendi.",
    "İşe geç kaldım, patron biraz kızdı.",
    "Dün gece çok geç yattım, uykum var.",
    "Yeni diziyi tek sezonda bitirdim.",
    "Bugün herkes çok sessizdi.",
    "Kargom hala gelmedi, merak ediyorum.",
    "Yemek siparişim yanlış geldi.",
    "Dersler bu dönem biraz daha zor.",
    "Yolda yürürken eski bir arkadaşımı gördüm.",
    "Bugün diyet yapmaya başladım.",
    "Ev çok dağınık, biraz toparlamam lazım.",
    "Sınav ertelendi, biraz rahatladım.",
    "Kafede oturup kitap okumak çok keyifli.",
    "Bugün güneş gözlüğümü unuttum.",
    "Markette yeni indirimler vardı.",
    "Arkadaşlarımla buluşmak moralimi düzeltti.",
    "Pencereyi açınca içeri güzel bir koku doldu.",
    "Yeni aldığım kitap hemen bitti.",
    "Bugün moralim çok yüksekti.",
    "Sokakta çocuklar top oynuyordu, izledim.",
    "Akşam yemeği için dışarı çıkmayı düşünüyorum."
]



# %%  text cleaning and preprocessing:tokenization,padding, label encoding

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset)
total_words = len(tokenizer.word_index) + 1 #☺ total kelime sayısı

# ngram dizi oluştur ve padding uygula

input_sequence= []
for text in dataset:
    #metinleri kelimelere çevir
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # her metin için ngram dizi oluştur
    for i in range(1,len(token_list)):
        n_gram_sequence= token_list[:i+1]
        input_sequence.append(n_gram_sequence)

# en uzun dizinin len bul ve hepsini aynı len yap
max_sequence_length = max(len(x) for x in input_sequence)

# dizilere padding uygula, hepsini aynı uzunlukta olmasini sağla

input_sequence = pad_sequences(input_sequence,
                             maxlen=max_sequence_length,
                             padding="pre")


# input and target 
# tüm satırlar ve tüm sütünların -1'incisine kadar
x = input_sequence[:,:-1]
y= input_sequence[:,-1]

y = tf.keras.utils.to_categorical(y,num_classes=total_words) # one hot encoding

# %%  LSTM model , compile ,train and evaluate

model = Sequential()

# x normalde (407,7) ya 1. indexi alınca 7 yi almış oluyoruz
model.add(Embedding(total_words,
                    50,
                    input_length = x.shape[1]))

# LSTM
#100 nörön sayısı
# false olması sonuncuyu döndermesi sadece
model.add(LSTM(100,
               return_sequences=False))

model.add(Dense(total_words,
                activation = "softmax"))

# compile

model.compile(optimizer ="adam",
              loss="categorical_crossentropy",
              metrics= ["accuracy"])

model.fit(x,y,epochs=100,verbose=1)



# %% model prediction

def generate_text(seed_text,next_words):
    # kaç kelimeyse o kadar eklesin diye loop
    for _ in range(next_words):
        #convert input to numerical
        
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        #padding
        token_list = pad_sequences([token_list],
                                   maxlen=max_sequence_length-1,
                                   padding="pre")
        
        #prediction
        # vebose = 0 consolada göstermez
        # verbose = 1 consolade gösterir accuracy oranı vs.
        predicted_probabilities = model.predict(token_list,
                                                verbose=0)
        
        # burda 300 kadar predict edecek 
        # en yüksek olasılığa sahip olanı al
        
        # burda sayıları bulduk
        predicted_word_index = np.argmax(predicted_probabilities,axis=-1)
        
        
        #şimdi tokenizer ile kelime indexinden asıl kelime bulunur
        
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        
        # tahmin edilen kelimeyi seed text e ekle
        
        seed_text = seed_text +" "+predicted_word
        
    return seed_text



seed_text ="Kitap okumak "
print(generate_text(seed_text, 4))            



































