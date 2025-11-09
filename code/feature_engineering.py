import numpy as np
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    
    def __init__(self, data: pd.DataFrame, sectors_data: pd.DataFrame, 
                 lookback_window: int = 60):
        self.data = data.copy()
        self.sectors_data = sectors_data.copy()
        self.lookback_window = lookback_window
        self.features = None
    
    def compute_momentum_features(self) -> pd.DataFrame:
        # Momentum features (Jegadeesh & Titman 1993)
        mom_features = pd.DataFrame(index=self.sectors_data.index)
        
        for col in self.sectors_data.columns:
            mom_features[f'{col}_MOM20'] = self.sectors_data[col].pct_change(20)
            mom_features[f'{col}_MOM60'] = self.sectors_data[col].pct_change(60)
            spy_mom20 = self.sectors_data['SPY'].pct_change(20)
            mom_features[f'{col}_REL_MOM20'] = mom_features[f'{col}_MOM20'] - spy_mom20
        
        return mom_features
    
    def compute_volatility_features(self) -> pd.DataFrame:
        # Volatility features (Ang & Bekaert 2002)
        vol_features = pd.DataFrame(index=self.sectors_data.index)
        
        for col in self.sectors_data.columns:
            ret = self.sectors_data[col].pct_change()
            vol_features[f'{col}_HV30'] = ret.rolling(30).std() * np.sqrt(252)
            vol_features[f'{col}_HV60'] = ret.rolling(60).std() * np.sqrt(252)
        
        if 'VIX' in self.data.columns:
            vix_mean = self.data['VIX'].rolling(60).mean()
            vix_std = self.data['VIX'].rolling(60).std()
            vol_features['VIX_ZSCORE'] = (self.data['VIX'] - vix_mean) / (vix_std + 1e-8)
            vol_features['VIX_HIGH_VOL'] = (self.data['VIX'] > vix_mean).astype(float)
            vol_features['VIX_MOM20'] = self.data['VIX'].pct_change(20)
        
        return vol_features
    
    def compute_mean_reversion_features(self) -> pd.DataFrame:
        # Mean reversion indicators (De Bondt & Thaler 1985)
        mr_features = pd.DataFrame(index=self.sectors_data.index)
        
        for col in self.sectors_data.columns:
            ret = self.sectors_data[col].pct_change()
            ret_mean = ret.rolling(60).mean()
            ret_std = ret.rolling(60).std()
            mr_features[f'{col}_RET_ZSCORE'] = (ret - ret_mean) / (ret_std + 1e-8)
            
            bb_mid = self.sectors_data[col].rolling(20).mean()
            bb_std = self.sectors_data[col].rolling(20).std()
            bb_width = 2 * bb_std
            mr_features[f'{col}_BB_POSITION'] = (
                (self.sectors_data[col] - (bb_mid - bb_width)) / (2 * bb_width + 1e-8)
            ).clip(0, 1)
        
        return mr_features
    
    def compute_yield_curve_features(self) -> pd.DataFrame:
        # Yield curve features (Estrella & Mishkin 1998)
        yc_features = pd.DataFrame(index=self.data.index)
        
        if 'SPREAD_10Y_13W' in self.data.columns:
            spread = self.data['SPREAD_10Y_13W']
            yc_features['SPREAD_LEVEL'] = spread
            yc_features['SPREAD_MOM20'] = spread.diff(20)
            
            spread_mean = spread.rolling(120).mean()
            spread_std = spread.rolling(120).std()
            yc_features['SPREAD_ZSCORE'] = (spread - spread_mean) / (spread_std + 1e-8)
            yc_features['CURVE_INVERTED'] = (spread < 0).astype(float)
        
        if 'TNX' in self.data.columns:
            yc_features['TNX_MOM20'] = self.data['TNX'].diff(20)
            yc_features['TNX_ZSCORE'] = (
                (self.data['TNX'] - self.data['TNX'].rolling(120).mean()) / 
                (self.data['TNX'].rolling(120).std() + 1e-8)
            )
        
        return yc_features
    
    def compute_uncertainty_features(self) -> pd.DataFrame:
        # Macro uncertainty features
        unc_features = pd.DataFrame(index=self.data.index)
        
        if 'EPU' in self.data.columns:
            epu_mean = self.data['EPU'].rolling(60).mean()
            epu_std = self.data['EPU'].rolling(60).std()
            unc_features['EPU_ZSCORE'] = (self.data['EPU'] - epu_mean) / (epu_std + 1e-8)
            unc_features['EPU_SPIKE'] = (self.data['EPU'] > (epu_mean + 2*epu_std)).astype(float)
        
        if 'GPR' in self.data.columns:
            gpr_mean = self.data['GPR'].rolling(60).mean()
            gpr_std = self.data['GPR'].rolling(60).std()
            unc_features['GPR_ZSCORE'] = (self.data['GPR'] - gpr_mean) / (gpr_std + 1e-8)
            unc_features['GPR_SPIKE'] = (self.data['GPR'] > (gpr_mean + 2*gpr_std)).astype(float)
        
        if 'EPU' in self.data.columns and 'GPR' in self.data.columns:
            unc_features['UNCERTAINTY_COMPOSITE'] = (
                (self.data['EPU'] / self.data['EPU'].rolling(60).mean()) +
                (self.data['GPR'] / self.data['GPR'].rolling(60).mean())
            ) / 2
        
        return unc_features
    
    def compute_macro_features(self) -> pd.DataFrame:
        # Macro fundamental features
        macro_features = pd.DataFrame(index=self.data.index)
        
        for col in ['USD', 'OIL', 'GOLD']:
            if col in self.data.columns:
                macro_features[f'{col}_MOM20'] = self.data[col].pct_change(20)
                col_mean = self.data[col].rolling(60).mean()
                col_std = self.data[col].rolling(60).std()
                macro_features[f'{col}_ZSCORE'] = (self.data[col] - col_mean) / (col_std + 1e-8)
        
        return macro_features
    
    def engineer_all_features(self) -> pd.DataFrame:
        momentum = self.compute_momentum_features()
        volatility = self.compute_volatility_features()
        mean_reversion = self.compute_mean_reversion_features()
        yield_curve = self.compute_yield_curve_features()
        uncertainty = self.compute_uncertainty_features()
        macro = self.compute_macro_features()
        
        all_features = pd.concat([
            self.data,
            momentum,
            volatility,
            mean_reversion,
            yield_curve,
            uncertainty,
            macro
        ], axis=1)
        
        self.features = all_features
        return all_features
    
    def get_features_info(self) -> Dict[str, list]:
        if self.features is None:
            return {"error": "Call engineer_all_features() first"}
        
        base_features = list(self.data.columns)
        all_cols = set(self.features.columns)
        engineered_cols = list(all_cols - set(base_features))
        
        return {
            "base_features": base_features,
            "total_engineered": len(engineered_cols),
            "engineered_features": sorted(engineered_cols),
            "total_features": len(self.features.columns)
        }
