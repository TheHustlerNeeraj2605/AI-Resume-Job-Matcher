import time
import os
import psutil
import pandas as pd
from sentence_transformers import CrossEncoder

print("=" * 70)
print("PHASE 7: CROSS-ENCODER CPU FEASIBILITY BENCHMARK")
print("=" * 70)

# Load candidate_train sample
train_df = pd.read_csv('data/processed/candidate_train.csv')
sample_pairs = list(zip(train_df['resume_text'].iloc[:100], train_df['job_description_text'].iloc[:100]))

print(f"Loaded 100 sample pairs from candidate_train.")

# Measure model load time
t0 = time.time()
print("Loading model: cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
t_load = time.time() - t0
print(f"Model loaded in {t_load:.2f}s.")

# Memory before benchmark
proc = psutil.Process(os.getpid())
mem_before = proc.memory_info().rss / (1024 * 1024)

# Measure inference time on 100 pairs with batch_size=32
t1 = time.time()
scores = model.predict(sample_pairs, batch_size=32, show_progress_bar=False)
t_infer = time.time() - t1

mem_after = proc.memory_info().rss / (1024 * 1024)
peak_mem = mem_after

avg_latency_ms = (t_infer / len(sample_pairs)) * 1000

print(f"\n--- BENCHMARK RESULTS ---")
print(f"Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
print(f"Batch size: 32")
print(f"Processed: {len(sample_pairs)} pairs")
print(f"Total benchmark time: {t_infer:.2f}s")
print(f"Average latency per pair: {avg_latency_ms:.2f} ms")
print(f"Estimated throughput: {len(sample_pairs)/t_infer:.1f} pairs/sec")
print(f"Process Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB (Peak delta: {peak_mem - mem_before:+.1f} MB)")

# Estimate full dataset costs
splits = {
    'candidate_train': len(pd.read_csv('data/processed/candidate_train.csv')),
    'candidate_validation': len(pd.read_csv('data/processed/candidate_validation.csv')),
    'candidate_test': len(pd.read_csv('data/processed/candidate_test.csv')),
    'official_test': len(pd.read_csv('data/raw/test.csv'))
}

total_pairs = sum(splits.values())
est_total_seconds = (total_pairs * avg_latency_ms) / 1000

print(f"\n--- DATASET RUNTIME ESTIMATES ---")
for split_name, count in splits.items():
    est_sec = (count * avg_latency_ms) / 1000
    print(f"  {split_name:<22}: {count:>5} pairs -> {est_sec:>6.1f}s ({est_sec/60:>4.1f} min)")
print(f"  {'TOTAL':<22}: {total_pairs:>5} pairs -> {est_total_seconds:>6.1f}s ({est_total_seconds/60:>4.1f} min)")

latency_df = pd.DataFrame([{
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'loading_time_s': round(t_load, 2),
    'benchmark_pairs': len(sample_pairs),
    'benchmark_time_s': round(t_infer, 2),
    'avg_latency_per_pair_ms': round(avg_latency_ms, 2),
    'throughput_pairs_per_sec': round(len(sample_pairs)/t_infer, 1),
    'memory_mb': round(peak_mem, 1),
    'est_total_pairs': total_pairs,
    'est_total_time_min': round(est_total_seconds / 60, 2)
}])
os.makedirs('data/evaluation', exist_ok=True)
latency_df.to_csv('data/evaluation/phase7_cross_encoder_latency.csv', index=False)
print(f"\nSaved data/evaluation/phase7_cross_encoder_latency.csv")
print("Feasibility benchmark complete!")
