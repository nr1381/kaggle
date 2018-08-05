# アニメデータからk近傍法を利用し、特定のアニメのタイトルを入力するとレコメンドする機能

# 1.前処理
## データ入力
ratings = pd.read_csv('/Users/ryonakay/develop/kaggle/anime/rating.csv')
anime = pd.read_csv('/Users/ryonakay/develop/kaggle/anime/anime.csv')

## anime_id同士で二つのテーブルを結合
join = pd.merge(ratings_tmp, anime_tmp, on='anime_id')

## 必要なデータのみ利用
available_join = join[['name', 'user_id', 'rating_x']]

## ratingが-1のものは削除
available_join = available_join[available_join['rating_x'] != -1]

## アニメ名をindexにし、カラムをuser_id、値をratingにテーブル変形
pivot_join = available_join.pivot_table(index= 'name',columns='user_id',values='rating_x')

## Nan値を0に置換
pivot_join =  pivot_join.fillna(0)

## スパースな行列なためcsr_matrix形式に変換
## スパースな行列は特殊な行列なため、これらの形式にすると高速になる
## http://hamukazu.com/2014/09/26/scipy-sparse-basics/
csr = csr_matrix(pivot_join.values)

# 2.学習
## k近傍法により学習
## k近傍法は引数が1つのものと2つのもの2つある １つのものは教師なし 2つのものは教師ありだがk近傍法は教師なしなはず?
knn = NearestNeighbors(n_neighbors=9,algorithm= 'brute', metric= 'cosine')
model = knn.fit(csr)

# 3.予測
## ハンガレンを読んでいる人に10のアニメを推薦する
Anime = 'Fullmetal Alchemist'
distance, indice = model.kneighbors(pivot_join.iloc[pivot_join.index== Anime].values.reshape(1,-1),n_neighbors=11)
pivot_join.index[indice]

