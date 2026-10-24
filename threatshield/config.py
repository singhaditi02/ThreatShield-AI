"""
config.py  ·  Section 2 of the notebook
Schema, attack-class mapping, encoders, model factory and class descriptions.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ── Schema (identical to the original notebook) ────────────────
ALL_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
]

ATTACK_MAP = {
    "normal":"normal","back":"dos","buffer_overflow":"u2r","ftp_write":"r2l","guess_passwd":"r2l",
    "imap":"r2l","ipsweep":"probe","land":"dos","loadmodule":"u2r","multihop":"r2l","neptune":"dos",
    "nmap":"probe","perl":"u2r","phf":"r2l","pod":"dos","portsweep":"probe","rootkit":"u2r",
    "satan":"probe","smurf":"dos","spy":"r2l","teardrop":"dos","warezclient":"r2l","warezmaster":"r2l",
}

PROTO_MAP = {"icmp": 0, "tcp": 1, "udp": 2}
FLAG_MAP  = {"SF":0,"S0":1,"REJ":2,"RSTR":3,"RSTO":4,"SH":5,"S1":6,"S2":7,"RSTOS0":8,"S3":9,"OTH":10}

# Columns removed in the original pipeline (correlation / constant / high-cardinality)
DROP_COLS = ["num_root","srv_serror_rate","srv_rerror_rate","dst_host_srv_serror_rate",
             "dst_host_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
             "dst_host_same_srv_rate","is_host_login","num_outbound_cmds","service"]
FEATURES = [c for c in ALL_COLUMNS if c not in DROP_COLS]          # 30 model features
CATEGORICAL = ["protocol_type", "flag"]

CLASS_INFO = {
    "normal": dict(name="Normal Traffic", sev="SAFE",     color="#22c55e",
                   desc="Traffic matches legitimate behaviour patterns. No action required."),
    "dos":    dict(name="Denial of Service (DoS)", sev="HIGH", color="#f97316",
                   desc="Flooding behaviour intended to exhaust bandwidth, memory or connection tables (e.g. neptune, smurf, back, teardrop)."),
    "probe":  dict(name="Probing / Reconnaissance", sev="MEDIUM", color="#eab308",
                   desc="Network or port scanning to map hosts and discover vulnerable services (e.g. nmap, portsweep, ipsweep, satan)."),
    "r2l":    dict(name="Remote-to-Local (R2L)", sev="HIGH", color="#ef4444",
                   desc="Unauthorised remote access attempts such as password guessing or FTP abuse (e.g. guess_passwd, warezclient, ftp_write)."),
    "u2r":    dict(name="User-to-Root (U2R)", sev="CRITICAL", color="#dc2626",
                   desc="Privilege-escalation attempts from a normal account to root (e.g. buffer_overflow, rootkit, loadmodule)."),
}

MODEL_FACTORY = {
    "Naive Bayes":         lambda: GaussianNB(),
    "Decision Tree":       lambda: DecisionTreeClassifier(criterion="entropy", max_depth=4),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=0),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(random_state=0),
    "SVM":                 lambda: SVC(gamma="scale"),
}
SLOW_MODELS = {"SVM", "Gradient Boosting"}
CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
