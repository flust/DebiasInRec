# Explaination

## grid folder

- Purpose: Output 5% **ground truth model** for filtering
- Process:
    - Link necessary files:
        - item file: item.ffm
        - user file: rd.ffm(5%) rd.tr.ffm(2.5%) rd.va.ffm(2.5%)
    - Config grid-rd.sh to grid on 2.5%tr/2.5va ground truth data
    - ./grid-rd.sh
    - Save binary model on **5% data** with best parameter and iteration


## filter folder

- Purpose: Use **ground truth model** to filter data
- Process:
    - Link necessary files:
        - user file (90% to be filtered): fl.ffm
        - user file (5% once used to train **ground truth model**): rf.ffm
        - item file: item.ffm
        - model: model
    - Config run-filter.sh
    - ./run-filter.sh


## add-pos-bias folder

- Purpos: Add position bias to modify filtered datas
- Process:
    - Link necessary files:
        - user file (90%): fl.ffm
        - filtered data: determined_filter propensious_filter random_filter
    - Config add_bais_all.sh
    - ./add_bias_all.sh
    - Config merge.sh
    - ./merge.sh
