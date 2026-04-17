print("start import")
import time

time1 = time.time()
from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from BTNHGV2HeteroDataClass import BTNHGV2HeteroDataClass
from HeteroModelTrainerTesterClass import HeteroModelTrainerTesterClass
from HGTClass import HGTClass
from HeteroGCNClass import HeteroGCNClass

time2 = time.time()
print("import used time: ", time2 - time1)
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")

# 一键切换实验：只改这里
EXPERIMENT_NAME = "heterogcn_a"

EXPERIMENTS = {
	# 公平对比同参组（A）
	"hgt_a": {
		"model": "hgt",
		"hidden_channels": 64,
		"num_layers": 3,
		"dropout": 0.2,
		"lr": 0.005,
		"weight_decay": 0.0005,
	},
	"heterogcn_a": {
		"model": "heterogcn",
		"hidden_channels": 64,
		"num_layers": 3,
		"dropout": 0.2,
		"lr": 0.005,
		"weight_decay": 0.0005,
	},
	# 轻量调参组（B/C/D）
	"heterogcn_b": {
		"model": "heterogcn",
		"hidden_channels": 64,
		"num_layers": 2,
		"dropout": 0.2,
		"lr": 0.01,
		"weight_decay": 0.0005,
	},
	"heterogcn_c": {
		"model": "heterogcn",
		"hidden_channels": 64,
		"num_layers": 3,
		"dropout": 0.3,
		"lr": 0.005,
		"weight_decay": 0.0005,
	},
	"heterogcn_d": {
		"model": "heterogcn",
		"hidden_channels": 128,
		"num_layers": 2,
		"dropout": 0.2,
		"lr": 0.005,
		"weight_decay": 0.0005,
	},
}

if EXPERIMENT_NAME not in EXPERIMENTS:
	raise ValueError(f"未知实验名: {EXPERIMENT_NAME}")

cfg = EXPERIMENTS[EXPERIMENT_NAME]
print(f"Running experiment: {EXPERIMENT_NAME}")
print(cfg)

# 处理数据集
heteroDataCls = BTNHGV2HeteroDataClass()

# 定义模型
if cfg["model"] == "hgt":
	gmodel = HGTClass(
		heteroData=heteroDataCls.heteroData,
		hidden_channels=cfg["hidden_channels"],
		out_channels=BTNHGV2ParameterClass.out_channels,
		num_heads=BTNHGV2ParameterClass.num_heads,
		num_layers=cfg["num_layers"],
		dropout=cfg["dropout"],
		useProj=BTNHGV2ParameterClass.HGT_useProj,
	)
elif cfg["model"] == "heterogcn":
	gmodel = HeteroGCNClass(
		heteroData=heteroDataCls.heteroData,
		hidden_channels=cfg["hidden_channels"],
		num_layers=cfg["num_layers"],
		dropout=cfg["dropout"],
	)
else:
	raise ValueError(f"不支持的模型: {cfg['model']}")

# 准备训练器测试器
trainertester = HeteroModelTrainerTesterClass(
	model=gmodel,
	lr=cfg["lr"],
	weight_decay=cfg["weight_decay"],
	epochs=BTNHGV2ParameterClass.epochs,
	patience=BTNHGV2ParameterClass.patience,
	useTrainWeight=BTNHGV2ParameterClass.useTrainWeight,
	min_delta=BTNHGV2ParameterClass.min_delta,
	folderPath=BTNHGV2ParameterClass.dataPath,
	resultFolderName=BTNHGV2ParameterClass.resultFolderName,
	kFold_k=BTNHGV2ParameterClass.kFold_k,
	batch_size=BTNHGV2ParameterClass.batch_size,
	useLrScheduler=BTNHGV2ParameterClass.useLrScheduler,
	epochsDisplay=BTNHGV2ParameterClass.epochsDisplay,
)

# 交叉验证
resultAnalyCls = trainertester.kFold_train_test()
resultAnalyCls.save_kFold()
