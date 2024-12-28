import mongoose from "mongoose";

const DefaultConfigSchema = new mongoose.Schema({
  input_folder: { type: String, required: true },
  output_folder: { type: String, required: true },
  input_table_name: { type: String, required: true },
  oot_table_name: { type: String, required: true },
  target_column: { type: String, required: true },
  Index: { type: String, required: true },
  task_type: { type: String, required: true },
  train_test_ratio: { type: Number, required: true },
  local_file: {
    use: { type: Boolean, required: true },
    train_file_name: { type: String, required: true },
    oot_file_name: { type: String, required: true },
  },
  snowflake_connection: {
    account: { type: String, required: true },
    user: { type: String, required: true },
    password: { type: String, required: true },
    warehouse: { type: String, required: true },
    database: { type: String, required: true },
    schema: { type: String, required: true },
  },
  data_preparation: {
    enabled: { type: String, required: true },
    fillna: { type: String, required: true },
    fill_value: { type: String, required: true },
    dropna: { type: String, required: true },
    standardize: { type: String, required: true },
    label_encoding: { type: String },
    categorical_columns: { type: [String], required: true },
  },
  eda: {
    enabled: { type: String, required: true },
  },
  feature_exploration: {
    enabled: { type: String, required: true },
    model_type: { type: String, required: true },
    task_type: { type: String, required: true },
    feature_selection: {
      top_n: { type: Number, required: true },
      model_for_selection: { type: String, required: true },
    },
    feature_imp_plot: { type: String, required: true },
  },
  build: {
    enabled: { type: String, required: true },
    cv_enabled: { type: Boolean, required: true },
    cv_folds: { type: Number, required: true },
    cumulative_importance_threshold: { type: Number, required: true },
    models: {
      linear_regression: { type: Object },
      random_forest: { type: Object },
      xgboost: { type: Object },
    },
  },
  score: {
    enabled: { type: String, required: true },
  },
});

const DefaultConfig = mongoose.model("DefaultConfig",DefaultConfigSchema)

export default DefaultConfig

