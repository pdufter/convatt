# path to penn treebank
/mounts/work/philipp/data/penn_treebank_3/ # from LDC or rather local copy of it
cp /mounts/data/corp/nltk-data/corpora/ptb/allcats.txt ~/nltk_data/corpora/ptb/

ln -s /mounts/work/philipp/data/penn_treebank_3/parsed/mrg/wsj/ ~/nltk_data/corpora/ptb/WSJ
ln -s /mounts/work/philipp/data/penn_treebank_3/parsed/mrg/brown/ ~/nltk_data/corpora/ptb/BROWN

# uppercase all folders
cd /mounts/work/philipp/data/penn_treebank_3/parsed/mrg/wsj/
find . -depth | \
    while read LONG 
    do 
        SHORT=$( basename "$LONG" | tr '[:lower:]' '[:upper:]' )
        DIR=$( dirname "$LONG" ) 
        if [ "${LONG}" != "${DIR}/${SHORT}"  ] 
        then 
            mv "${LONG}" "${DIR}/${SHORT}" 
        fi 
    done
cd /mounts/work/philipp/data/penn_treebank_3/parsed/mrg/brown/
find . -depth | \
    while read LONG 
    do 
        SHORT=$( basename "$LONG" | tr '[:lower:]' '[:upper:]' )
        DIR=$( dirname "$LONG" ) 
        if [ "${LONG}" != "${DIR}/${SHORT}"  ] 
        then 
            mv "${LONG}" "${DIR}/${SHORT}" 
        fi 
    done
