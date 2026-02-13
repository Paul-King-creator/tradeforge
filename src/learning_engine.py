"""
Learning Engine - Aus Trades lernen und Strategien optimieren
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List
import statistics

class LearningEngine:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
    
    def analyze_winning_patterns(self) -> Dict:
        """Muster in erfolgreichen Trades finden"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Gewinn-Trades analysieren
        cursor.execute('''
            SELECT strategy, AVG(confidence), AVG(pnl_percent),
                   setup_description
            FROM trades 
            WHERE status = 'closed' AND pnl_percent > 0
            GROUP BY strategy
        ''')
        
        winning_patterns = {}
        for row in cursor.fetchall():
            strategy, avg_conf, avg_pnl, setup = row
            winning_patterns[strategy] = {
                'avg_confidence': avg_conf,
                'avg_return': avg_pnl,
                'sample_setup': setup
            }
        
        # Verlust-Trades analysieren
        cursor.execute('''
            SELECT strategy, AVG(confidence), AVG(pnl_percent)
            FROM trades 
            WHERE status = 'closed' AND pnl_percent <= 0
            GROUP BY strategy
        ''')
        
        losing_patterns = {}
        for row in cursor.fetchall():
            strategy, avg_conf, avg_pnl = row
            losing_patterns[strategy] = {
                'avg_confidence': avg_conf,
                'avg_loss': avg_pnl
            }
        
        conn.close()
        
        return {
            'winning': winning_patterns,
            'losing': losing_patterns
        }
    
    def get_strategy_recommendations(self) -> List[str]:
        """Empfehlungen basierend auf Performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT strategy, 
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl_percent) as avg_return
            FROM trades 
            WHERE status = 'closed'
            GROUP BY strategy
            HAVING COUNT(*) >= 5
            ORDER BY avg_return DESC
        ''')
        
        recommendations = []
        for row in cursor.fetchall():
            strategy, total, wins, avg_ret = row
            win_rate = (wins / total * 100) if total > 0 else 0
            
            if win_rate > 55 and avg_ret > 0:
                recommendations.append(
                    f"✅ {strategy}: Starke Performance ({win_rate:.0f}% Win Rate, Ø {avg_ret:.2f}%)"
                )
            elif win_rate < 40:
                recommendations.append(
                    f"⚠️ {strategy}: Schwache Performance ({win_rate:.0f}% Win Rate) - Überarbeiten oder deaktivieren"
                )
        
        conn.close()
        return recommendations
    
    def calculate_optimal_parameters(self) -> Dict:
        """Optimale Parameter aus historischen Trades berechnen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Beste Risk/Reward Ratio finden
        cursor.execute('''
            SELECT 
                (take_profit - entry_price) / NULLIF(entry_price - stop_loss, 0) as rr_ratio,
                pnl_percent
            FROM trades 
            WHERE status = 'closed' AND signal_type = 'buy'
        ''')
        
        rr_data = cursor.fetchall()
        
        # Nach RR-Ratio gruppieren
        rr_performance = {}
        for rr, pnl in rr_data:
            if rr is None:
                continue
            rr_bucket = round(rr * 2) / 2  # Runden auf 0.5 Schritte
            if rr_bucket not in rr_performance:
                rr_performance[rr_bucket] = []
            rr_performance[rr_bucket].append(pnl)
        
        # Beste RR-Ratio finden
        best_rr = None
        best_avg_pnl = float('-inf')
        
        for rr, pnls in rr_performance.items():
            if len(pnls) >= 3:  # Mindestens 3 Trades
                avg_pnl = statistics.mean(pnls)
                if avg_pnl > best_avg_pnl:
                    best_avg_pnl = avg_pnl
                    best_rr = rr
        
        conn.close()
        
        return {
            'optimal_risk_reward': best_rr,
            'expected_return_at_optimal': best_avg_pnl if best_avg_pnl > float('-inf') else 0,
            'sample_size_by_rr': {k: len(v) for k, v in rr_performance.items()}
        }
    
    def generate_learning_report(self) -> str:
        """Lern-Report generieren"""
        patterns = self.analyze_winning_patterns()
        recommendations = self.get_strategy_recommendations()
        optimal_params = self.calculate_optimal_parameters()
        
        report = [
            "=" * 60,
            "🧠 TRADEFORGE LEARNING REPORT",
            "=" * 60,
            "",
            "📈 STRATEGIE EMPFEHLUNGEN",
            "-" * 40,
        ]
        
        if recommendations:
            report.extend(recommendations)
        else:
            report.append("Noch nicht genug Daten für Empfehlungen (mind. 5 Trades pro Strategie)")
        
        report.extend([
            "",
            "⚙️ OPTIMALE PARAMETER",
            "-" * 40,
            f"Optimale Risk/Reward Ratio: {optimal_params['optimal_risk_reward']}:1" 
            if optimal_params['optimal_risk_reward'] else "Noch nicht genug Daten",
            "",
            "🎯 GEWINN MUSTER",
            "-” * 40,
        ])
        
        for strategy, data in patterns['winning'].items():
            report.append(
                f"\n{strategy}:"
                f"\n  Ø Konfidenz: {data['avg_confidence']:.0%}"
                f"\n  Ø Return: {data['avg_return']:.2f}%"
            )
        
        report.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(report)