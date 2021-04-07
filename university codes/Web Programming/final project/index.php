<?php
session_start();
$signed_in = false;
$good5Image_dir = '';
$good5Name = '';
$good5Price = 0;
if (isset($_SESSION['username']) && !empty($_SESSION['username'])) {
    $signed_in = true;
}
//extract categories from database
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

$inner5Goods = '';
$good5Image_dir = '';
$good5Name = '';
$good5Price = 0;
$good5Counter = 0;

$sql = "SELECT id,name,price,image_dir FROM good 
order by id desc";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        if ($good5Counter >= 5) {
            break;
        }
//        $categories .= $row["categoryName"];
        $good5Image_dir = $row["image_dir"];
        $good5Image_dir = substr($good5Image_dir, 1);
//        echo $good5Image_dir;
        $good5Name = $row["name"];
        $good5Price = $row["price"];
        $inner5Goods .= '<div class="lastGoods" onclick="formPost(' . $row["id"] . ')" id="';
        if ($good5Counter % 2 == 0) {
            $inner5Goods .= 'even';
        } else {
            $inner5Goods .= 'odd';
        }
        $good5Counter++;
        $inner5Goods .= '">
<img class="good5" src="' . $good5Image_dir . '">
<br>
نام محصول:
' . $good5Name . '
<br>
<br>
قیمت محصول:
' . $good5Price . '
تومان
<br>

</div>';
    }
}


$categories = "";
$sql = "SELECT categoryName FROM categories";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $categories .= "<a onclick=\"myPostForCategory('" . $row["categoryName"] . "')\" class=\"catOrStore\" >";
        $categories .= $row["categoryName"];
        $categories .= '</a>';
    }
}

$stores = "";
$sql = "SELECT name FROM store";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $stores .= "<a onclick=\"myPostForStore('" . $row["name"] . "')\" class=\"catOrStore\" >";
        $stores .= $row["name"];
        $stores .= '</a>';
    }
}

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>digikala</title>
    <link rel="stylesheet" href="./css/first.css">
</head>
<body>
<form method="post" action="./php/goodByCategory.php" id="goodByCategoryForm" name="goodByCategoryForm">
</form>
<form method="post" action="./php/goodByStore.php" id="goodByStoreForm" name="goodByStoreForm">
</form>
<div id="top">
    <div id="top_right">
        <img class="image" src="./images/digikala.png">
        <div id="search_div">
            <img class="image" id="search" src="./images/search.PNG" onclick="search()">
            <input id="search_text" type="text" value="نام کالا فروشگاه یا دسته موردنظر..."/>
        </div>
    </div>
    <div id="top_left">';
//$string .= '<img class="topleft" src="./images/basket.PNG" onclick="go_to_basket()">';


if (!$signed_in) {
    $string .= '<img class="topleft" src="./images/basket.PNG" onclick="window.alert(\'برای اضافه کردن کالا به سبد خرید وارد حساب کاربری خود شوید.\')">
 <img class="topleft" src="./images/sign_in.PNG" onclick="go_to_register()">';
} else {
    $string .= '<img class="topleft" src="./images/basket.PNG" onclick="go_to_basket()">
<div id="sign_in" onclick="sign_out()">' . $_SESSION['username'] . '</div>';
}
$string .= '</div>
</div>
<div class="navbar">
    <div class="dropdown">
        <button class="dropbtn">دسته بندی کالاها
            <i class="fa fa-caret-down"></i>
        </button>
        <div class="dropdown-content">
            ' . $categories . '
        </div>
    </div>
    <div class="dropdown">
        <button class="dropbtn">فروشگاه ها
            <i class="fa fa-caret-down"></i>
        </button>
        <div class="dropdown-content">
            ' . $stores . '
        </div>
    </div>
    <a href="./php/sellerRegister.php">فروشنده شوید</a>
</div>
<div id="content_div">

<div id="title">
<h3>آخرین محصولات ثبت شده</h3>
</div>

<form method="post" action="./php/buyGood.php" id="5Goods" name="5Goods">
</form>
<div id="last5Goods">

' . $inner5Goods . '

</div>

</div>




</div>
<script src="./js/main_page.js"></script>
</body>
</html>';

echo $string;
?>
