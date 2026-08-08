import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

N_VENDORS = 50
N_ITEMS = 100
N_WAREHOUSES = 10
N_POS = 800
N_INVOICES = 700
N_INVENTORY = 500


def random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_purchase_orders() -> pd.DataFrame:
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    records = []

    for po_id in range(1, N_POS + 1):
        order_date = random_date(start_date, end_date)
        expected_days = random.randint(7, 60)
        expected_delivery = order_date + timedelta(days=expected_days)

        status = random.choices(
            ["delivered", "delayed", "cancelled"],
            weights=[0.65, 0.25, 0.10],
        )[0]

        if status == "delivered":
            actual_delivery = expected_delivery + timedelta(days=random.randint(-5, 10))
        elif status == "delayed":
            actual_delivery = expected_delivery + timedelta(days=random.randint(5, 45))
        else:
            actual_delivery = None

        records.append({
            "po_id": po_id,
            "vendor_id": random.randint(1, N_VENDORS),
            "item_id": random.randint(1, N_ITEMS),
            "quantity": random.randint(1, 500),
            "unit_price": round(random.uniform(5.0, 500.0), 2),
            "order_date": order_date.strftime("%Y-%m-%d"),
            "expected_delivery": expected_delivery.strftime("%Y-%m-%d"),
            "actual_delivery": actual_delivery.strftime("%Y-%m-%d") if actual_delivery else None,
            "status": status,
        })

    df = pd.DataFrame(records)

    # Inject ~10% additional missing on actual_delivery for delivered rows
    delivered_mask = df["status"] == "delivered"
    noise = np.random.rand(len(df)) < 0.10
    df.loc[delivered_mask & noise, "actual_delivery"] = None

    return df


def generate_invoices(po_ids: list) -> pd.DataFrame:
    sampled_po_ids = random.sample(list(po_ids), min(N_INVOICES, len(po_ids)))
    records = []

    for inv_id, po_id in enumerate(sampled_po_ids, 1):
        amount_due = round(random.uniform(100.0, 50000.0), 2)

        # Class imbalance: 60% paid, 25% unpaid, 15% partial
        status = random.choices(
            ["paid", "unpaid", "partial"],
            weights=[0.60, 0.25, 0.15],
        )[0]

        if status == "paid":
            amount_paid = amount_due
            days_overdue = 0
            payment_date = fake.date_between(start_date="-2y", end_date="today").strftime("%Y-%m-%d")
        elif status == "unpaid":
            amount_paid = 0.0
            days_overdue = random.randint(1, 180)
            payment_date = None
        else:
            amount_paid = round(random.uniform(0.1, 0.9) * amount_due, 2)
            days_overdue = random.randint(0, 90)
            payment_date = (
                fake.date_between(start_date="-2y", end_date="today").strftime("%Y-%m-%d")
                if random.random() > 0.3 else None
            )

        records.append({
            "invoice_id": inv_id,
            "po_id": po_id,
            "amount_due": amount_due,
            "amount_paid": amount_paid,
            "payment_date": payment_date,
            "days_overdue": days_overdue,
            "payment_status": status,
        })

    df = pd.DataFrame(records)

    # Inject ~10-12% realistic missingness on financial columns
    for col in ["amount_paid", "days_overdue"]:
        mask = np.random.rand(len(df)) < 0.11
        df.loc[mask, col] = np.nan

    return df


def generate_inventory() -> pd.DataFrame:
    records = []
    seen = set()

    while len(records) < N_INVENTORY:
        item_id = random.randint(1, N_ITEMS)
        warehouse_id = random.randint(1, N_WAREHOUSES)
        if (item_id, warehouse_id) in seen:
            continue
        seen.add((item_id, warehouse_id))

        reorder_point = random.randint(10, 200)
        stock_level = random.randint(0, 1000)
        last_restocked = fake.date_between(start_date="-1y", end_date="today").strftime("%Y-%m-%d")

        records.append({
            "item_id": item_id,
            "warehouse_id": warehouse_id,
            "stock_level": stock_level,
            "reorder_point": reorder_point,
            "last_restocked": last_restocked,
        })

    df = pd.DataFrame(records)

    # ~15% missing on last_restocked, ~10% on reorder_point
    df.loc[np.random.rand(len(df)) < 0.15, "last_restocked"] = None
    df.loc[np.random.rand(len(df)) < 0.10, "reorder_point"] = np.nan

    return df


if __name__ == "__main__":
    print("Generating supply chain dataset...")

    po_df = generate_purchase_orders()
    po_df.to_csv("purchase_orders.csv", index=False)
    print(f"purchase_orders.csv  — {len(po_df)} rows")

    inv_df = generate_invoices(po_df["po_id"].tolist())
    inv_df.to_csv("invoices.csv", index=False)
    print(f"invoices.csv         — {len(inv_df)} rows")
    print(f"  payment_status distribution:\n{inv_df['payment_status'].value_counts().to_string()}")

    inventory_df = generate_inventory()
    inventory_df.to_csv("inventory.csv", index=False)
    print(f"inventory.csv        — {len(inventory_df)} rows")

    print("\nTarget columns for ML:")
    print("  Classification : payment_status  (invoices.csv)")
    print("  Regression     : days_overdue    (invoices.csv)")
