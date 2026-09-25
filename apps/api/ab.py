"""Asignación A/B determinista (copia exacta de credit_risk.monitoring.ab_testing.assign_variant).

La app se despliega sola (sin el paquete del repo), por eso se duplica esta
función; un test en CI verifica que ambas asignen igual.
"""

import hashlib


def assign_variant(applicant_id: str, challenger_traffic: float, salt: str = "ab-v1") -> str:
    if challenger_traffic <= 0:
        return "champion"
    digest = hashlib.sha256(f"{salt}:{applicant_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "challenger" if bucket < challenger_traffic else "champion"
