<?php
session_start();

$servername = "localhost";
$user = "root";
$password = "";
$dbname = "final_project";

// Create connection
$conn = new mysqli($servername, $user, $password, $dbname);
mysqli_set_charset($conn, "utf8");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $buying_id = $_POST["id"];

    $sql = "DELETE FROM buying where id=" . $buying_id;
    $result = $conn->query($sql);

}


$person_id = 0;


$sql = "SELECT id FROM person where email=\"" . $_SESSION['email'] . "\"";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $person_id = $row["id"];
    }
}

$innerBody = '';

$buying_id = 0;
$image_dir = '';
$goodName = '';
$price = 0;
$category = '';
$storeName = '';
$description = '';
$sellerId = 0;
$totalPrice = 0;

$sql = "SELECT id,good_id FROM buying where status =0 and person_id =" . $person_id;
$result = $conn->query($sql);

if ($result->num_rows > 0) {
// output data of each row
    while ($row = $result->fetch_assoc()) {
        $buying_id = $row["id"];
        $sql2 = "SELECT name,categoryName,image_dir,price,store_id,description FROM good where id=" . $row["good_id"];
        $result2 = $conn->query($sql2);
        if ($result2->num_rows > 0) {
            // output data of each row
            while ($row2 = $result2->fetch_assoc()) {
                $image_dir = $row2["image_dir"];
                $goodName = $row2["name"];
                $price = $row2["price"];
                $totalPrice = $totalPrice + $price;
                $category = $row2["categoryName"];
                $description = $row2["description"];
                $sellerId = $row2["store_id"];
                $sql3 = "SELECT name from store where id=" . $sellerId;
                $result3 = $conn->query($sql3);
                if ($result3->num_rows > 0) {
                    // output data of each row
                    while ($row3 = $result3->fetch_assoc()) {
                        $storeName = $row3["name"];
                    }
                }

                $innerBody .= '<div class="main">
    <div class="image">
        <img src="' . $image_dir . '">
    </div>
    <div class="center">
        <br>
        نام محصول:
        ' . $goodName . '
        <br>
        <br>
        قیمت محصول:
        ' . $price . '
        تومان
        <br>
        <br>
        دسته بندی:
        ' . $category . '
        <br>
        <br>
        نام فروشگاه:
        ' . $storeName . '
        <br>
        <br>
        توضیحات:
        ' . $description . '
    </div>
    <div class="left">
        <br>
        <br>
        <br>
        <br>
        <form method="post" action="./basket.php">
            <input type="number" value=' . $buying_id . ' name="id" style="display: none;">
            <button class="noConfirmBtn">حذف از سبد خرید</button>
        </form>
        <br>
        <br>
        <br>
        <br>
    </div>
</div>
<div class="space"></div>';

            }
        }
    }
}

$conn->close();
//
$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>تقاضای ثبت محصول</title>
    <link rel="stylesheet" href="../css/confirm.css">
</head>
<body>
' . $innerBody . '
<div id="total">
    <p id="totalText">جمع قیمت کل:</p>
    <p id="totalPrice"> ' . $totalPrice . ' تومان</p>
</div>
<div id="finish">
        <button id="buyComplete" onclick="buyComplete()">اتمام خرید</button>
</div>
<script src="../js/basket.js"></script>
</body>
</html>';

echo $string;

?>

