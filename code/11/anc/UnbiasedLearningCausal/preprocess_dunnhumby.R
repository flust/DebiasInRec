

## parameters
by_category <- TRUE
path_dir <- "data/raw"
type_recommendation <- "mailer"

filter_data_by_promotion_existence <- TRUE
filter_data_by_active_store <- FALSE
min_time_user <- 10
min_time_item <- 10
min_time_user_treatment <- 1
min_time_item_treatment <- 1
min_item_treated_outcome <- 1
min_item_control_outcome <- 1

## load library
library(data.table)

## functions
# Convert from item_id to new_id in dt (drop rows which)
# INPUT
# dt: dataframe to be coverted; include original item_id column
# ref: reference table for conversion; include item_id and new_id columns
# OUTPUT
# dt: after the conversion
func_conv_item_id <- function(dt, ref){
  dt <- dt[item_id %in% ref$item_id]
  setkey(ref, "item_id")
  setkey(dt, "item_id")
  dt <- merge(dt, ref)
  dt[, item_id := NULL]
  setnames(dt, "new_id", "item_id")
  return(dt)
}

func_print_log <- function(x, with_time=TRUE){
  if(with_time){
    print(paste(Sys.time(), x)) 
  }else{
    print(x)
  }
}


# Prepare data from Dunnhumby datasets.
# INPUT
# path_dir: directory of datasets.
# type_recommendation: select type of recommendation (mailer/display).
# by_category: choose items' granularity from either SUB_COMMODITY_DESC (by_category=TRUE) or PRODUCT_ID (by_category=FALSE)."
# filter_data_by_promotion_existence: if TRUE, only select stores with transaction logs of whole weeks (default is TRUE).
# filter_data_by_active_store: if TRUE, only select stores with transaction logs of whole weeks (default is TRUE).
# OUTPUT
# list of four dataframes: 
# hist_outcomes (time_id, user_id, item_id, quantity, sales)
# hist_treatments (time_id, user_id, item_id, quantity, sales)
# ref_users (user_id, "features")
# ref_items (item_id, "features")

func_prepare_dunnhumby <- function(path_dir, min_time_user=5, min_time_item=5, min_time_user_treatment=1, min_time_item_treatment=1,
                                   min_item_treated_outcome=1, min_item_control_outcome=1,
                                  type_recommendation="mailer", by_category=FALSE, 
                                  filter_data_by_promotion_existence=TRUE, filter_data_by_active_store=TRUE){
  # read data
  func_print_log("Read data.")
  transaction_data <- fread(paste0(path_dir, "/", "transaction_data.csv"))
  causal_data <- fread(paste0(path_dir, "/", "causal_data.csv"))
  hh_demographic <- fread(paste0(path_dir, "/", "hh_demographic.csv"))
  product <- fread(paste0(path_dir, "/", "product.csv"))
  
  # temp <- transaction_data[,.N, by=WEEK_NO][order(WEEK_NO)]
  # head(temp, 100)
  # sort(unique(transaction_data$WEEK_NO)) 1-102
  # sort(unique(causal_data$WEEK_NO)) 9-101
  
  # extend user data
  func_print_log("Extend user data.")
  unique_user <- transaction_data[, .(household_key = sort(unique(household_key)))]
  setkey(unique_user, "household_key")
  setkey(hh_demographic, "household_key")
  hh_demographic <- merge(unique_user, hh_demographic, all.x=T)
  
  # choose type of recommendation
  func_print_log("Choose type of recommendation.")
  if(type_recommendation == "mailer"){
    func_print_log("type_recommendation = mailer", with_time=F)
    causal_data <- causal_data[mailer != "0"]
  }else if(type_recommendation == "display"){
    func_print_log("type_recommendation = display", with_time=F)
    causal_data <- causal_data[display != "0"]
  }
  
  # change col names
  func_print_log("Change col names.")
  setnames(transaction_data, "WEEK_NO", "time_id")
  setnames(causal_data, "WEEK_NO", "time_id")
  
  setnames(transaction_data, "household_key", "user_id")
  setnames(hh_demographic, "household_key", "user_id")
  
  setnames(transaction_data, "PRODUCT_ID", "item_id")
  setnames(causal_data, "PRODUCT_ID", "item_id")
  setnames(product, "PRODUCT_ID", "item_id")
  
  if(by_category){
    func_print_log("Items' granularity is SUB_COMMODITY_DESC.", with_time=F)
    ref <- product[COMMODITY_DESC != "NO COMMODITY DESCRIPTION" & SUB_COMMODITY_DESC != "NO SUBCOMMODITY DESCRIPTION", 
                     .(item_id, new_id = paste(COMMODITY_DESC, SUB_COMMODITY_DESC, sep="::"))]
    transaction_data <- func_conv_item_id(transaction_data, ref)
    causal_data <- func_conv_item_id(causal_data, ref)
    product <- func_conv_item_id(product, ref)
  }else{
    func_print_log("Items' granularity is PRODUCT_ID", with_time=F)
  }
  
  # sum(is.na(causal_data$item_id))
  # pre_transaction_data <- transaction_data[time_id < min(causal_data[, unique(time_id)])]
  # only select times, stores, items with promotion logs
  if(filter_data_by_promotion_existence){
    func_print_log("Only select times, stores, items with promotion logs.")
    func_print_log(paste0("Original total times: ", length(unique(transaction_data$time_id))), with_time=F)
    func_print_log(paste0("Times with promotion logs: ", length(unique(causal_data$time_id))), with_time=F)
    func_print_log(paste0("Original total stores: ", length(unique(transaction_data$STORE_ID))), with_time=F)
    func_print_log(paste0("Stores with promotion logs: ", length(unique(causal_data$STORE_ID))), with_time=F)
    func_print_log(paste0("Original total items: ", length(unique(transaction_data$item_id))), with_time=F)
    func_print_log(paste0("Items with promotion logs: ", length(unique(causal_data$item_id))), with_time=F)
    
    func_print_log(paste0("Original data volume: ", nrow(transaction_data)), with_time=F)
    transaction_data <- transaction_data[time_id %in% causal_data[, unique(time_id)]]
    causal_data <- causal_data[STORE_ID %in% transaction_data[, unique(STORE_ID)]]
    transaction_data <- transaction_data[STORE_ID %in% causal_data[, unique(STORE_ID)]]
    causal_data <- causal_data[item_id %in% transaction_data[, unique(item_id)]]
    transaction_data <- transaction_data[item_id %in% causal_data[, unique(item_id)]]
    func_print_log(paste0("Trimmed data volume: ", nrow(transaction_data)), with_time=F)
  }
  
  # only select stores with transaction logs of whole weeks
  if(filter_data_by_active_store){
    func_print_log("Only select stores with transaction logs of whole weeks.")
    temp <- transaction_data[, .(N = .N, cnt_week = length(unique(time_id))), by = STORE_ID]
    active_store <- temp[cnt_week == max(temp$cnt_week)]
    
    func_print_log(paste0("Original total stores: ", length(unique(transaction_data$STORE_ID))), with_time=F)
    func_print_log(paste0("Stores with promotion logs: ", length(unique(active_store$STORE_ID))), with_time=F)
    
    func_print_log(paste0("Original data volume: ", nrow(transaction_data)), with_time=F)
    transaction_data <- transaction_data[STORE_ID %in% active_store[, unique(STORE_ID)]]
    func_print_log(paste0("Trimmed data volume: ", nrow(transaction_data)), with_time=F)
  }
  
  # filter user, item, promotion information by the existence on transaction data
  func_print_log("Filter user, item, promotion information by the existence on transaction data.")

  cnt_time_items <- transaction_data[, .(cnt = length(unique(time_id))), by = item_id]
  cnt_time_users <- transaction_data[, .(cnt = length(unique(time_id))), by = user_id]

  # cnt_time_items[,.N, by = cnt >= 10]
  # cnt_time_users[,.N, by = cnt >= 10]
  product <- product[item_id %in% cnt_time_items[cnt >= min_time_item, item_id]]
  hh_demographic <- hh_demographic[user_id %in% cnt_time_users[cnt >= min_time_user, user_id]]
  
  transaction_data <- transaction_data[user_id %in% hh_demographic$user_id & item_id %in% product$item_id]
  causal_data <- causal_data[STORE_ID %in% transaction_data[, unique(STORE_ID)] & item_id %in% transaction_data[, unique(item_id)]]
  
  func_print_log(paste0("Users with at least ", min_time_user, " weeks transaction: ", length(unique(transaction_data$user_id))), with_time=F)
  func_print_log(paste0("Items with at least ", min_time_item, " weeks transaction: ", length(unique(transaction_data$item_id))), with_time=F)
  
  func_print_log("Prepare outcomes (time_id, user_id, item_id, quantity, sales).")
  
  
  # hist_outcomes <- unique(transaction_data[, .(time_id, user_id, item_id)])
  hist_outcomes <- transaction_data[, .(quantity = sum(QUANTITY), sales = sum(SALES_VALUE)), by = c("time_id", "user_id", "item_id")]
  # hist_outcomes_pre <- pre_transaction_data[, .(quantity = sum(QUANTITY), sales = sum(SALES_VALUE)), by = c("time_id", "user_id", "item_id")]
  
  # prepare treatments
  func_print_log("Prepare treatments (time_id, user_id, item_id).")
  func_print_log("Extract store visits.")
  # visit_data <- unique(transaction_data[time_id %in% causal_data[, unique(time_id)] & STORE_ID %in% causal_data[, unique(STORE_ID)], 
  #                                       .(STORE_ID, time_id, user_id)])
  visit_data <- unique(transaction_data[time_id %in% causal_data[, unique(time_id)] & STORE_ID %in% causal_data[, unique(STORE_ID)], 
                                        .(STORE_ID, time_id, user_id)])
  func_print_log("Extract promotion of each store.")
  promotion_data <- unique(causal_data[,.(STORE_ID, time_id, item_id)])
  setkey(visit_data, "STORE_ID", "time_id")
  setkey(promotion_data, "STORE_ID", "time_id")
  
  # sum(is.na(promotion_data$item_id))
  func_print_log("Merge (outer join) store visits of users and store promotions of items.")
  hist_treatments <- merge(promotion_data, visit_data, all = T, allow.cartesian=TRUE)
  hist_treatments <- hist_treatments[!is.na(user_id) & !is.na(item_id)]
  
  func_print_log("Remove duplicate promotions from some visited stores.")
  hist_treatments <- unique(hist_treatments[, .(time_id, item_id, user_id)])[order(time_id, item_id, user_id)]
  
  # check again the existence of treatment data
  cnt_time_items_treatment <- hist_treatments[, .(cnt = length(unique(time_id))), by = item_id]
  cnt_time_users_treatment <- hist_treatments[, .(cnt = length(unique(time_id))), by = user_id]
  
  product <- product[item_id %in% cnt_time_items_treatment[cnt >= min_time_item_treatment, item_id]]
  hh_demographic <- hh_demographic[user_id %in% cnt_time_users_treatment[cnt >= min_time_user_treatment, user_id]]
  hist_outcomes <- hist_outcomes[item_id %in% product[, unique(item_id)] & user_id %in% hh_demographic[, unique(user_id)]]
  hist_treatments <- hist_treatments[item_id %in% product[, unique(item_id)] & user_id %in% hh_demographic[, unique(user_id)]]
  
  func_print_log("Add treatment info to hist_outcomes.")
  hist_treatments$treated <- 1
  setkey(hist_outcomes, "time_id", "user_id", "item_id")
  setkey(hist_treatments, "time_id", "user_id", "item_id")
  hist_outcomes <- merge(hist_outcomes[,.(time_id, user_id, item_id)], hist_treatments, all.x = T)
  hist_outcomes[is.na(treated), treated := 0]
  
  #
  cnt_item_treated_outcome <- hist_outcomes[treated == 1, .(cnt = .N), by = item_id]
  cnt_item_control_outcome <- hist_outcomes[treated == 0, .(cnt = .N), by = item_id]
  product <- product[item_id %in% cnt_item_treated_outcome[cnt >= min_item_treated_outcome, item_id] &
                       item_id %in% cnt_item_control_outcome[cnt >= min_item_control_outcome, item_id]]
  hist_outcomes <- hist_outcomes[item_id %in% product[, unique(item_id)] & user_id %in% hh_demographic[, unique(user_id)]]
  hist_treatments <- hist_treatments[item_id %in% product[, unique(item_id)] & user_id %in% hh_demographic[, unique(user_id)]]
  
  func_print_log("Completed.")
  
  func_print_log(paste0("Number of users: ", nrow(hh_demographic)), with_time=F)
  func_print_log(paste0("Number of items: ", length(unique(product$item_id))), with_time=F)
  
  return(list(hist_outcomes=hist_outcomes, hist_treatments=hist_treatments, 
              ref_users=hh_demographic, ref_items=product))
  # return(list(hist_outcomes=hist_outcomes, hist_treatments=hist_treatments, 
  #             ref_users=hh_demographic, ref_items=product, hist_outcomes_pre=hist_outcomes_pre))
}

func_cnt_logs <- function(hist_outcomes, hist_treatments, ref_users, ref_items){
  unique_users <- sort(unique(ref_users$user_id))
  unique_items <- sort(unique(ref_items$item_id))
  cnt_logs <- data.table(user_id = rep(unique_users, length(unique_items)),
                         item_id = rep(unique_items, each = length(unique_users)))
 
  
  func_print_log("Count important numbers.")
  cnt_visits <- hist_treatments[, .(num_visit = length(unique(time_id))), by = "user_id"]
  cnt_outcomes <- hist_outcomes[, .(num_outcome = .N), by = c("user_id", "item_id")]
  cnt_treated_outcomes <- hist_outcomes[treated == 1, .(num_treated_outcome = .N), by = c("user_id", "item_id")]
  cnt_treatments <- hist_treatments[, .(num_treatment = .N), by = c("user_id", "item_id")]
  
  func_print_log("Merge important numbers.")
  # user_id, item_id, num_visit, num_treatment, num_outcome, num_treated_outcome
  setkey(cnt_treatments, "user_id", "item_id")
  setkey(cnt_treated_outcomes, "user_id", "item_id")
  setkey(cnt_outcomes, "user_id", "item_id")
  setkey(cnt_logs, "user_id", "item_id")
  cnt_logs <- merge(cnt_logs, cnt_treatments, all.x = T, allow.cartesian=TRUE)
  cnt_logs <- merge(cnt_logs, cnt_outcomes, all.x = T, allow.cartesian=TRUE)
  cnt_logs <- merge(cnt_logs, cnt_treated_outcomes, all.x = T)
  setkey(cnt_visits, "user_id")
  setkey(cnt_logs, "user_id")
  cnt_logs <- merge(cnt_logs, cnt_visits, all.x = T)
  func_print_log("Fill NA with 0.")
  # func_print_log(cnt_logs[is.na(num_treatment), .N])
  # func_print_log(cnt_logs[is.na(num_outcome), .N])
  # func_print_log(cnt_logs[is.na(num_treated_outcome), .N])
  cnt_logs[is.na(num_treatment), num_treatment := 0]
  cnt_logs[is.na(num_outcome), num_outcome := 0]
  cnt_logs[is.na(num_treated_outcome), num_treated_outcome := 0]
  # func_print_log("Calculate causal effect.")
  # cnt_logs[, causal_effect := num_treated_outcome/num_treatment - (num_outcome - num_treated_outcome)/(num_visit - num_treatment)]
  
  return(cnt_logs)
}

func_conv_index <- function(cnt_logs){
  unique_users <- sort(unique(cnt_logs$user_id))
  unique_items <- sort(unique(cnt_logs$item_id))
  func_print_log(paste0("Number of unique users: ", length(unique_users)), with_time=F)
  func_print_log(paste0("Number of unique items: ", length(unique_items)), with_time=F)
  cnt_logs$idx_user <- as.integer(factor(cnt_logs$user_id, levels = unique_users)) - 1
  cnt_logs$idx_item <- as.integer(factor(cnt_logs$item_id, levels = unique_items)) - 1
  return(cnt_logs)
}



## execute
# category level
by_category <- TRUE

t <- proc.time() # 
list_dataset <- func_prepare_dunnhumby(path_dir, min_time_user, min_time_item, 
                                       min_time_user_treatment, min_time_item_treatment,
                                       min_item_treated_outcome, min_item_control_outcome,
                                       type_recommendation, by_category, 
                                       filter_data_by_promotion_existence, filter_data_by_active_store)

cnt_logs <- func_cnt_logs(list_dataset$hist_outcomes, list_dataset$hist_treatments,
                          list_dataset$ref_users, list_dataset$ref_items)
cnt_logs <- func_conv_index(cnt_logs)
proc.time() - t


if(by_category){
  save_name <- paste0("data/preprocessed/dunn_cat_", type_recommendation, 
                      "_", min_time_user, "_", min_time_item, "_", min_time_user_treatment, "_", min_time_item_treatment, 
                      "/cnt_logs.csv")
}else{
  save_name <- paste0("data/preprocessed/dunn_", type_recommendation, 
                      "_", min_time_user, "_", min_time_item, "_", min_time_user_treatment, "_", min_time_item_treatment,  "/cnt_logs.csv")
}
if(!dir.exists(dirname(save_name))){
  dir.create(dirname(save_name))
}
write.csv(cnt_logs[, .(idx_user, idx_item, num_visit, num_treatment, num_outcome, num_treated_outcome)], 
          save_name, row.names=F)

# product level
by_category <- FALSE
t <- proc.time() # 
list_dataset <- func_prepare_dunnhumby(path_dir, min_time_user, min_time_item, 
                                       min_time_user_treatment, min_time_item_treatment,
                                       min_item_treated_outcome, min_item_control_outcome,
                                       type_recommendation, by_category, 
                                       filter_data_by_promotion_existence, filter_data_by_active_store)

cnt_logs <- func_cnt_logs(list_dataset$hist_outcomes, list_dataset$hist_treatments,
                          list_dataset$ref_users, list_dataset$ref_items)
cnt_logs <- func_conv_index(cnt_logs)
proc.time() - t


if(by_category){
  save_name <- paste0("data/preprocessed/dunn_cat_", type_recommendation, 
                      "_", min_time_user, "_", min_time_item, "_", min_time_user_treatment, "_", min_time_item_treatment, 
                      "/cnt_logs.csv")
}else{
  save_name <- paste0("data/preprocessed/dunn_", type_recommendation, 
                      "_", min_time_user, "_", min_time_item, "_", min_time_user_treatment, "_", min_time_item_treatment,  "/cnt_logs.csv")
}
if(!dir.exists(dirname(save_name))){
  dir.create(dirname(save_name))
}
write.csv(cnt_logs[, .(idx_user, idx_item, num_visit, num_treatment, num_outcome, num_treated_outcome)], 
          save_name, row.names=F)
