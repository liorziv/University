import urllib
import json
import mechanize

# all the urls i crwlad - since the web blocked me i tried like this and it worked
urls = ["https://www.walmart.com/browse/online-specials-toys/?page=2&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=3&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=4&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=5&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=6&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=7&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=8&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=9&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=10&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=11&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=12&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=13&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=14&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=15&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=16&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=17&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=18&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=19&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=20&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=21&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=22&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=23&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=24&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=25&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=26&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=27&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=28&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=29&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=30&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=31&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=32&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=33&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=34&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=35&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=36&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=37&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=38&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=39&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=40&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=41&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=42&cat_id=0",
        "https://www.walmart.com/browse/online-specials-toys/?page=43&cat_id=0",
"https://www.walmart.com/browse/online-specials-toys/?page=44&cat_id=0" ,
"https://www.walmart.com/browse/online-specials-toys/?page=45&cat_id=0"]

#will hold all the items
itemsList = []

#start to pass on each items page
for url in urls:

    # turns the page into an html source page format
    htmlFileContant = mechanize.urlopen(url).read()

    #the labels to look up
    lablels = [("data-item-id" ,"data-seller-id"), ("<a class=js-product-image " ,"> <img class"), ("baseprice=", "dohideitemcontrols"), ("<span class=stars-reviews>", "<")] # ("</span> <a class=js-product-image href="," > <img class")] #""<a class=js-product-image href=", "baseprice=" ]



    initial = 0
    start_idx = 0
    end_idx = 0
    details = [0]*4

    #will continue searching all the products in the page untill it is finished
    while(start_idx != -1 and end_idx != -1):
        counter = 0
        for (label,end) in (lablels):
            start_idx = htmlFileContant.find(label,initial)
            end_idx = htmlFileContant.find(end, start_idx)
            initial = end_idx
            details[counter] = htmlFileContant[start_idx+len(label):end_idx]
            counter +=1
        #builds an item as a dict for the json file
        item = {'url' : url ,'data-item-id': details[0] ,'image': details[1] , 'baseprice': details[2], 'stars-reviews' : details[3]}

        #appends each items
        itemsList.append(item);



#turns the list into a dict
dict = {"items" : itemsList}

#the name of ourput file
fileName = "wallmartJONS"
jsonFile = open(fileName, 'w')

#convert the items dict into JSON file format
jsonFile.write(json.dumps(dict, indent = 4 , separators= (",", ": ") ))
jsonFile.close()
