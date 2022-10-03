# item-based CF
import graphlab
train_data = graphlab.SFrame(df_CF)
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='Customer_id', item_id='item')
# Make Recommendations
item_sim_recomm = item_sim_model.recommend(users=['5208494361'],k=10)
item_sim_recomm.print_rows()