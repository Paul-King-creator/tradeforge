"""
Adaptive Strategy Optimizer
Passt Strategie-Parameter automatisch basierend auf Performance an
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics

class AdaptiveOptimizer:
    """
    Diese Klasse lernt aus Trades und passt Strategien automatisch an:
    
    1. Trackt Performance jeder Strategie
    2. Identifiziert erfolgreiche Parameter-Kombinationen
    3. Passt Schwellwerte an (z.B. RSI-Levels, Leverage)
    4. Deaktiviert schlechte Strategien temporär
    5. Aktiviert bessere Strategien stärker
    """
    
    def __init__(self, db_path: str = "data/trades.db", config_path: str = "config.yaml"):
        self.db_path = db_path
        self.config_path = config_path
        self.learning_rate = 0.1  # Wie schnell soll angepasst werden
        self.min_trades_for_adjustment = 10  # Mindestens 10 Trades pro Strategie
        
    def get_strategy_performance_history(self, strategy_name: str, days: int = 30) -> Dict:
        """Performance-Verlauf einer Strategie"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT 
                DATE(entry_time) as date,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_percent) as avg_pnl,
                AVG(confidence) as avg_confidence,
                AVG(leverage) as avg_leverage
            FROM trades 
            WHERE strategy = ? 
              AND status = 'closed'
              AND entry_time > ?
            GROUP BY DATE(entry_time)
            ORDER BY date DESC
        ''', (strategy_name, since))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'date': row[0],
                'trades': row[1],
                'wins': row[2],
                'win_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                'avg_pnl': row[3],
                'avg_confidence': row[4],
                'avg_leverage': row[5]
            })
        
        conn.close()
        return results
    
    def identify_best_parameters(self, strategy_name: str) -> Dict:
        """Finde die besten Parameter für eine Strategie"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Gewinn-Trades analysieren - welche Eigenschaften hatten sie?
        cursor.execute('''
            SELECT 
                leverage,
                confidence,
                entry_price,
                stop_loss,
                take_profit,
                pnl_percent
            FROM trades 
            WHERE strategy = ? 
              AND status = 'closed'
              AND pnl_percent > 0
            ORDER BY pnl_percent DESC
            LIMIT 20
        ''', (strategy_name,))
        
        winning_trades = cursor.fetchall()
        
        # Verlust-Trades analysieren
        cursor.execute('''
            SELECT 
                leverage,
                confidence,
                entry_price,
                stop_loss,
                take_profit,
                pnl_percent
            FROM trades 
            WHERE strategy = ? 
              AND status = 'closed'
              AND pnl_percent <= 0
            ORDER BY pnl_percent ASC
            LIMIT 20
        ''', (strategy_name,))
        
        losing_trades = cursor.fetchall()
        
        conn.close()
        
        # Beste Parameter ermitteln
        if not winning_trades:
            return {}
        
        avg_winning_leverage = statistics.mean([t[0] for t in winning_trades])
        avg_winning_confidence = statistics.mean([t[1] for t in winning_trades])
        
        # Risk/Reward Ratio der Gewinner
        winning_rr = []
        for t in winning_trades:
            entry, sl, tp = t[2], t[3], t[4]
            if entry and sl and tp and entry != sl:
                rr = abs(tp - entry) / abs(entry - sl)
                winning_rr.append(rr)
        
        avg_rr = statistics.mean(winning_rr) if winning_rr else 2.0
        
        return {
            'optimal_leverage': round(avg_winning_leverage, 1),
            'optimal_confidence_threshold': round(avg_winning_confidence - 0.05, 2),
            'optimal_risk_reward': round(avg_rr, 1),
            'sample_size': len(winning_trades)
        }
    
    def calculate_strategy_score(self, strategy_name: str) -> float:
        """
        Berechne einen Score für eine Strategie (0-100)
        Berücksichtigt: Win Rate, Profit Factor, Consistency
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Letzte 30 Tage
        since = (datetime.now() - timedelta(days=30)).isoformat()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_percent > 0 THEN pnl_percent ELSE 0 END) as total_gains,
                SUM(CASE WHEN pnl_percent <= 0 THEN ABS(pnl_percent) ELSE 0 END) as total_losses,
                AVG(pnl_percent) as avg_pnl
            FROM trades 
            WHERE strategy = ? 
              AND status = 'closed'
              AND entry_time > ?
        ''', (strategy_name, since))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] < 5:  # Zu wenig Daten
            return 50.0  # Neutraler Score
        
        total, wins, total_gains, total_losses, avg_pnl = row
        
        # Score-Komponenten
        win_rate_score = (wins / total * 100) if total > 0 else 0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else 1.0
        pf_score = min(profit_factor * 25, 50)  # Max 50 Punkte für PF
        
        # Konsistenz (weniger volatile = besser)
        consistency_score = 20 if win_rate_score > 40 else 10
        
        total_score = (win_rate_score * 0.4) + (pf_score * 0.4) + (consistency_score * 0.2)
        
        return round(min(max(total_score, 0), 100), 1)
    
    def get_strategy_adjustments(self) -> Dict:
        """
        Hauptmethode: Empfehlungen für Strategie-Anpassungen
        """
        adjustments = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'parameter_updates': {},
            'strategy_scores': {}
        }
        
        strategies = ['momentum_long', 'breakout_long', 'mean_reversion_long']
        
        for strategy in strategies:
            score = self.calculate_strategy_score(strategy)
            adjustments['strategy_scores'][strategy] = score
            
            # Performance History
            history = self.get_strategy_performance_history(strategy)
            
            if len(history) >= 3:  # Mindestens 3 Tage Daten
                recent_win_rate = statistics.mean([h['win_rate'] for h in history[:3]])
                
                # Empfehlungen basierend auf Score
                if score >= 70:
                    adjustments['recommendations'].append(
                        f"✅ {strategy}: Ausgezeichnet ({score}/100) - Erhöhe Position Size"
                    )
                elif score >= 50:
                    adjustments['recommendations'].append(
                        f"⚡ {strategy}: Gut ({score}/100) - Beibehalten"
                    )
                elif score >= 30:
                    adjustments['recommendations'].append(
                        f"⚠️  {strategy}: Schwach ({score}/100) - Reduziere Leverage"
                    )
                else:
                    adjustments['recommendations'].append(
                        f"🛑 {strategy}: Schlecht ({score}/100) - TEMPORÄR DEAKTIVIEREN"
                    )
                
                # Parameter-Optimierung bei genug Daten
                if sum(h['trades'] for h in history) >= self.min_trades_for_adjustment:
                    best_params = self.identify_best_parameters(strategy)
                    if best_params:
                        adjustments['parameter_updates'][strategy] = best_params
        
        return adjustments
    
    def apply_learning_to_config(self, config: Dict) -> Dict:
        """
        Wendet gelernte Optimierungen auf die Config an
        """
        adjustments = self.get_strategy_adjustments()
        
        updated_config = config.copy()
        
        # Leverage anpassen basierend auf Performance
        for strategy_name, params in adjustments.get('parameter_updates', {}).items():
            if strategy_name in updated_config.get('strategies', {}):
                strategy_config = updated_config['strategies'][strategy_name]
                
                # Nur anpassen wenn wir genug Daten haben
                if params.get('sample_size', 0) >= 10:
                    old_leverage = strategy_config.get('leverage', 3)
                    new_leverage = params['optimal_leverage']
                    
                    # Sanfte Anpassung (nicht zu drastisch)
                    adjusted_leverage = round((old_leverage * 0.7) + (new_leverage * 0.3))
                    adjusted_leverage = max(2, min(10, adjusted_leverage))  # Grenzen: 2-10x
                    
                    strategy_config['leverage'] = adjusted_leverage
                    
                    print(f"🎓 Learning: {strategy_name} Leverage {old_leverage}x → {adjusted_leverage}x")
        
        # Schlechte Strategien deaktivieren
        for strategy_name, score in adjustments['strategy_scores'].items():
            if score < 25 and strategy_name in updated_config.get('strategies', {}):
                updated_config['strategies'][strategy_name]['enabled'] = False
                print(f"🛑 Learning: {strategy_name} deaktiviert (Score: {score})")
        
        return updated_config
    
    def generate_learning_report(self) -> str:
        """Generiert einen Report für das Dashboard"""
        adjustments = self.get_strategy_adjustments()
        
        report_lines = [
            "=" * 60,
            "🧠 ADAPTIVE LEARNING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "📊 STRATEGY SCORES (0-100)",
            "-" * 40,
        ]
        
        for strategy, score in adjustments['strategy_scores'].items():
            emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            report_lines.append(f"{emoji} {strategy}: {score}/100")
        
        report_lines.extend([
            "",
            "💡 RECOMMENDATIONS",
            "-" * 40,
        ])
        
        for rec in adjustments['recommendations']:
            report_lines.append(rec)
        
        if adjustments['parameter_updates']:
            report_lines.extend([
                "",
                "⚙️ OPTIMAL PARAMETERS FOUND",
                "-" * 40,
            ])
            for strategy, params in adjustments['parameter_updates'].items():
                report_lines.append(f"\n{strategy}:")
                for key, value in params.items():
                    if key != 'sample_size':
                        report_lines.append(f"  • {key}: {value}")
        
        report_lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(report_lines)