<?php

$innerBody =$storeName =  '';

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

$exist = false;
$sql = "SELECT id,name,categoryName,image_dir,price,store_id,description FROM good where confirmed=0";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $sql2 = "SELECT name from store where id=" . $row["store_id"];
        $result2 = $conn->query($sql2);
        if ($result2->num_rows > 0) {
            // output data of each row
            while ($row2 = $result2->fetch_assoc()) {
                $storeName = $row2["name"];
            }
        }
        $innerBody .= '<div class="main">
    <div class="image">
        <img src="' . $row["image_dir"] . '">
    </div>
    <div class="center">
        <br>
        نام محصول:
        ' . $row["name"] . '
        <br>
        <br>
        قیمت محصول:
        ' . $row["price"] . '
        تومان
        <br>
        <br>
        دسته بندی:
        ' . $row["categoryName"] . '
        <br>
        <br>
        نام فروشگاه:
        ' . $storeName . '
        <br>
        <br>
        توضیحات:
        ' . $row["description"] . '
    </div>
    <div class="left">
        <br>
        <br>
        <button class="confirmBtn" onclick="confirm(' . $row["id"] . ')">تایید محصول</button>
        <br>
        <br>
        <br>
        <button class="noConfirmBtn" onclick="noConfirm(' . $row["id"] . ')">رد محصول</button>
        <br>
        <br>
        <br>
    </div>
</div>
<div class="space"></div>';
    }
    $innerBody .= '<script src="../js/confirm.js"></script>';
}


$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>تقاضای ثبت محصول</title>
    <link rel="stylesheet" href="../css/confirm.css">
</head>
<body>
' . $innerBody . '
</body>
</html>';

echo $string;


?>