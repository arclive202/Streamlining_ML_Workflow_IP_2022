#install package
#install.packages('completejourney')

#load package
library(completejourney)

#load data in the workspace
data(package = 'completejourney')

#get all the tables as dataframes
coupons <- as.data.frame(coupons)
demographics <- as.data.frame(demographics)
campaigns <- as.data.frame(campaigns)
campaign_descriptions <- as.data.frame(campaign_descriptions)
coupon_redemptions <- as.data.frame(coupon_redemptions)
products <- as.data.frame(products)
promotions_sample <- as.data.frame(promotions_sample)
transactions_sample <- as.data.frame(transactions_sample)
transactions <- get_transactions(verbose = FALSE)

#write the dataframes as csv files
write.csv(x=coupons, file="coupons.csv")
write.csv(x=demographics, file="demographics.csv")
write.csv(x=campaigns, file="campaigns.csv")
write.csv(x=campaign_descriptions, file="campaign_descriptions.csv")
write.csv(x=coupon_redemptions, file="coupon_redemptions.csv")
write.csv(x=products, file="products.csv")
write.csv(x=promotions_sample, file="promotions_sample.csv")
write.csv(x=transactions_sample, file="transactions_sample.csv")
write.csv(x=transactions, file="transactions.csv")


